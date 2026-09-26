# Architecture & Design Decisions

This document explains **why** RetinaEdge-DR is built the way it is. For how to run things,
see `docs/GETTING_STARTED.md` and `docs/TRAINING.md`.

## 1. Problem framing: ordinal grading, not softmax classification

Diabetic retinopathy (DR) severity is **ordinal**: the 5 ICDRSS grades
(No DR → Mild → Moderate → Severe → Proliferative) have a natural ordering, and the clinical
cost of confusing Mild with Moderate is far smaller than confusing Moderate with Proliferative.
A plain softmax cross-entropy throws this structure away and treats every misclassification
equally, which wastes capacity and produces badly calibrated neighbours.

RetinaEdge-DR therefore uses a **cumulative-link ordinal head** (CORAL-style,
Cao et al. 2020; see also Shi, Cao & Raschka 2021 for the CORN refinement):

- The head emits `K-1 = 4` logits, each solving the binary task `P(grade > k)`.
- Training uses BCE-with-logits on the binarised targets — each level is an easier,
  better-balanced problem than a 5-way softmax.
- Inference reconstructs per-grade probabilities as first differences of the sigmoid
  cumulatives (`retinaedge.models.ordinal_ops.ordinal_probs`), clamped and re-normalised
  for numerical safety.
- The **expected grade** `E[Y]` is available for severity trend analysis, and the
  **referable-DR probability** `P(grade ≥ 2)` is the clinically actionable output.

An auxiliary binary head for referable DR (`refer_logits`) is trained jointly
(`refer_weight=0.3` default). It acts as a regulariser that focuses representation
learning on the clinically critical threshold.

## 2. Backbone: MobileNetV3-Small first, EfficientNet-Lite0 second

Edge deployment on mid-range Android phones imposes a strict budget. Default targets:

| Budget item | Target |
|---|---|
| int8 model size | ≤ 5 MB |
| On-device latency (mid-range phone, 4 threads) | ≤ 150 ms |
| Server ONNX latency (2-core CPU) | ≤ 30 ms |

`timm.mobilenetv3_small_100` (~2.5 M params, ~60 M MACs @224px) fits with headroom; it is the
default in `configs/train/smoke.yaml` (smoke) and `configs/train/aptos_mobilenetv3.yaml`
(production). `efficientnet_lite0` (~4 M params) is the secondary configuration when more
accuracy is needed. Because every backbone is pulled from `timm` with `num_classes=0`,
any compatible backbone (e.g. `mobilenetv3_large_100`, `tf_efficientnet_lite1`) is a one-line
config change.

## 3. Preprocessing: Ben-Graham crop + probabilistic CLAHE

- **Ben-Graham preprocessing** (`retinaedge.data.ben_graham`) crops the fundus disc from the
  black background, rescales to a fixed radius (300 px) and applies CLAHE on the LAB
  L-channel. This is the standard preprocessing from the 2015 Kaggle DR competition winners
  and dramatically improves lesion visibility for weak cameras.
- **`clahe_prob = 0.5` is deliberate.** On-device (Android) we do **not** run CLAHE — it is
  too expensive in pure Kotlin without shipping OpenCV. By applying CLAHE randomly during
  training, the model becomes **invariant to its presence**, so the same exported graph
  works for on-device images (no CLAHE) and server-side images (CLAHE on).

## 4. Class imbalance

APTOS is severely imbalanced (roughly half of images are grade 0; proliferative cases are
~8%). Two complementary mechanisms:

1. **Class-balanced weighted sampler** (`train.sampler: true`) for the training loader.
2. **Focal weighting** of the ordinal BCE tasks (`focal_gamma: 2.0`) to down-weight easy
   negatives once the model is confident.

We deliberately do **not** oversample proliferative cases to parity — that destroys
calibration, which matters more for a screening tool than balanced confusion matrices.

## 5. Calibration as a first-class output

A screening model that says "85% referable" must mean it. Every run reports **ECE**
(15-bin expected calibration error) alongside QWK/AUC, and
`retinaedge.eval.calibration` fits **temperature scaling** (LBFGS on the val split NLL).
The fitted temperature is stored in `best.pt`, honoured by `predict_probs`, and written into
`model_info.json` so the Android app and the Gradio demo display calibrated probabilities.

## 6. Export contract (the Android handshake)

The exported graph is deliberately boring:

```
input : float32 (1, 3, H, W)   RGB, ImageNet-normalized (mean 0.485/0.456/0.406, std 0.229/0.224/0.225)
output: float32 (1, 5)         per-grade probabilities (sum = 1)
```

- Single output tensor → trivial to consume from TFLite/ONNX Runtime/Kotlin.
- Static spatial size, optional static batch for TFLite; dynamic batch for ONNX servers.
- `retinaedge.export.export_onnx` **verifies parity against PyTorch** (atol 1e-3) before the
  artifact is considered valid; CI runs this on every commit.
- int8 TFLite uses a representative dataset drawn from the validation transform pipeline
  (`retinaedge.export.representative`), so quantisation ranges see realistic activations.
- The Android reader handles both uint8-quantized and float32 I/O so either conversion path
  (`ai-edge-torch` direct, or `onnx2tf` fallback) works without app changes.

## 7. Why not bigger/fancier?

Ensembles, transformers (ViT/BEiT), and multi-scale architectures consistently add 1-3 AUC
points on DR benchmarks — and 10-50x the inference cost. The mission is a **screening-grade
model on a phone**, so the roadmap prioritises:

1. External validation (Messidor-2 / IDRiD / EyePACS-test) before any architecture change.
2. Quantisation-aware training (QAT) to recover int8 accuracy — cheaper than a bigger model.
3. Test-time augmentation flags in the *server* CLI only (TTA is a luxury phones don't have).

## References

- Cao, Mirjalili & Raschka. *Rank consistent ordinal regression for neural networks with
  application to age estimation.* Pattern Recognition Letters 140 (2020). (CORAL)
- Shi, Cao & Raschka. *Deep neural networks for rank-consistent ordinal regression based on
  conditional probabilities.* Pattern Analysis and Applications (2023). (CORN)
- Graham. *Kaggle Diabetic Retinopathy Detection competition report.* (2015) — Ben-Graham preprocessing.
- Gulshan et al. *Development and Validation of a Deep Learning Algorithm for Detection of
  Diabetic Retinopathy in Retinal Fundus Photographs.* JAMA 316(22) (2016).
- Ting et al. *Development and Validation of a Deep Learning System for Diabetic Retinopathy
  and Related Eye Diseases Using Retinal Images From Multiethnic Populations With Diabetes.* JAMA 318(22) (2017).
- Howard et al. *Searching for MobileNetV3.* ICCV (2019).
- Wu et al. *Machine learning quantization — AI Edge (ai-edge-torch).* Google (2024).
