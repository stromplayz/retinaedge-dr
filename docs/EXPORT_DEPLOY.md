# Export & Deployment

From a trained checkpoint to a quantized model running on a phone.
Owner modules: `retinaedge.export.*`. Consumer: the Android app (`com.retinaedge.app`).

## 1. The export graph contract (do not break)

Every exported artefact — ONNX or TFLite, float32 or int8 — honours:

```
input : float32 (1, 3, H, W), ImageNet-normalised
        mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)
output: float32 (1, 5) grade probabilities, rows sum to 1
        index = ICDRSS grade 0..4 (No DR … Proliferative DR)
```

- `H = W = cfg["data"]["img_size"]` (224 for real configs, 64 in smoke examples).
- The temperature fitted by calibration is **baked into the graph** — exported probs are already
  calibrated; consumers never rescale.
- Postprocess on consumer side: `grade = argmax(probs)`,
  `referable = probs[2] + probs[3] + probs[4] >= 0.5`.
- int8 models prefer **full-integer** quantization (uint8 in/out); the Android reader **must
  handle both uint8 and float32** IO.

## 2. PyTorch → ONNX

```bash
python -m retinaedge.export.export_onnx --config configs/train/aptos.yaml \
    --ckpt artifacts/aptos/best.pt --out artifacts/aptos/model.onnx \
    --img-size 224 --opset 17 --dynamic-batch
```

| Flag | Meaning |
|---|---|
| `--img-size` | override H/W (keep it equal to training unless you know better) |
| `--opset` | ONNX opset (default 17) |
| `--dynamic-batch` | allow `N` in the first dim (handy for batch eval on servers) |

`InferenceWrapper` (`retinaedge.export.wrappers`) wraps `DrNet` so the exported graph outputs
probs directly — no dict outputs, no K−1 logits leaking out.

**Parity check**: when `onnxruntime` is importable, the exporter runs both runtimes on a fixed
input and fails loudly above `atol 1e-3`; it also prints file size and a latency summary.

## 3. PyTorch → TFLite

Two paths, same graph contract:

```bash
# (a) direct: ai-edge-torch -> TFLite (preferred; needs the `export` extra)
python -m retinaedge.export.export_tflite --direct --config configs/train/aptos.yaml \
    --ckpt artifacts/aptos/best.pt --out artifacts/aptos/dr_model.tflite \
    --int8 --representative-dir data/aptos/images   # small folder of ~100-500 images

# (b) fallback: ONNX -> onnx2tf -> TFLite
python -m retinaedge.export.export_tflite --onnx artifacts/aptos/model.onnx \
    --out artifacts/aptos/dr_model.tflite --int8
```

int8 notes:

- Full-integer (uint8 in/out) quantization needs a **representative dataset** — point
  `--representative-dir` at a folder of preprocessed fundus images; never quantize with random
  noise if you can avoid it.
- Always re-run `evaluate`/`benchmark` on the int8 artefact; expect ≤ ~1% QWK drift and ~4×
  smaller files vs float32.

## 4. Benchmark

```bash
python -m retinaedge.export.benchmark --model artifacts/aptos/model.onnx \
    --backend onnxruntime --img-size 224 --runs 50
python -m retinaedge.export.benchmark --model artifacts/aptos/dr_model.tflite \
    --backend tflite --img-size 224 --runs 50
```

Reports mean/p95 latency and peak memory per inference. Rule-of-thumb targets on mid-range Android
(CPU, 4 threads): float32 TFLite ≈ 60–150 ms, int8 ≈ 25–60 ms at 224 px for a small backbone.

## 5. Metadata bundle

```bash
python -m retinaedge.export.metadata --model artifacts/aptos/dr_model.tflite \
    --out-dir artifacts/aptos/android --temperature 1.07
```

Writes the two files the Android app loads:

- `labels.txt` — exactly five lines, contract order:
  `No DR`, `Mild`, `Moderate`, `Severe`, `Proliferative DR`
- `model_info.json` — input size, normalisation constants, output shape, referable threshold,
  temperature, grade scheme documentation.

## 6. Android integration contract

The app (`android/`, owner: agent 2-d) consumes the contract above — nothing else:

| Item | Value |
|---|---|
| App id | `com.retinaedge.app` |
| Model path | `file:///android_asset/models/dr_model.tflite` |
| Labels path | `file:///android_asset/models/labels.txt` |
| Preprocess | crop fundus circle → resize 224 → **no CLAHE on device** → ImageNet normalise → quantize iff input tensor is uint8 |
| Postprocess | `grade = argmax(probs)`; `referable = probs[2..4].sum() ≥ 0.5` |

Why no CLAHE on device: training uses `clahe_prob: 0.5` randomisation precisely so the model is
robust with or without it (see [DATASETS.md](DATASETS.md#4-preprocessing)).

UI copy requirement: every result screen must carry the research-use disclaimer from the
[Model Card](MODEL_CARD.md) — this is a screening *support* demo, not a diagnostic device.

## 7. Sanity checklist before shipping a model

- [ ] ONNX parity vs PyTorch passed (`atol 1e-3`)
- [ ] int8 TFLite QWK on val within tolerance of float32
- [ ] `labels.txt` has exactly 5 lines in contract order
- [ ] `model_info.json` normalisation constants match §1
- [ ] benchmark numbers recorded (device, backend, img size, runs)
- [ ] temperature from calibration baked in and ECE re-measured
- [ ] disclaimer text present in the app UI
