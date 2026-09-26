# Model assets

This directory is where the exported TFLite graph is placed **at build time**.
Model binaries are never committed to git (`.gitignore` excludes `*.tflite`).

## Expected layout

```
app/src/main/assets/
├── labels.txt                 # committed fallback labels (ICDRSS order 0..4)
└── models/
    ├── dr_model.tflite        # REQUIRED — exported by the Python pipeline
    └── labels.txt             # optional, overrides ../labels.txt
```

## Producing the model

From the repo root (after `make setup` and a completed training run):

```bash
python -m retinaedge.export.export_tflite --direct \
    --config configs/train/smoke.yaml \
    --ckpt artifacts/smoke/best.pt \
    --out android/app/src/main/assets/models/dr_model.tflite

python -m retinaedge.export.metadata \
    --model android/app/src/main/assets/models/dr_model.tflite \
    --out-dir android/app/src/main/assets/models
```

## Contract the app depends on (docs/INTERFACES.md v1.0)

- Input: `float32 (1,3,H,W)` ImageNet-normalized
  (mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`).
- Output: `float32 (1,5)` grade probabilities over
  `0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR`.
- int8 models: full-integer with `uint8` in/out is supported — the app reads the
  tensor quantization parameters and quantizes/dequantizes automatically.
- NHWC layouts (e.g. via onnx2tf) are also auto-detected and handled.

If `models/dr_model.tflite` is missing, the app launches in a clearly-labelled
**demo mode** (synthetic probabilities) so the UI can be exercised without a
real checkpoint.
