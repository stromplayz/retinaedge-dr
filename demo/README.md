# Demo — Gradio web UI

Visual sanity check for trained RetinaEdge-DR models. **Research/educational use only — not a
medical device** (see [docs/MODEL_CARD.md](../docs/MODEL_CARD.md)).

## Run

```bash
pip install -e ".[demo]"
make demo                                  # = python demo/gradio_app.py
# explicit weights (priority: --onnx > --ckpt > artifacts/smoke/* > untrained fallback):
python demo/gradio_app.py --ckpt artifacts/smoke/best.pt
python demo/gradio_app.py --onnx artifacts/smoke/model.onnx --img-size 64
python demo/gradio_app.py --server-port 7861 --share   # public link
```

Open http://127.0.0.1:7860. If no checkpoint exists the app runs on **untrained fallback weights**
and says so in the report panel — the UI still works end-to-end.

## What it does

| Stage | Behaviour |
|---|---|
| Input | any fundus-like image (RGB/RGBA/grey), converted to uint8 RGB |
| Preprocess | optional Ben-Graham (crop+scale+CLAHE via `retinaedge.data.ben_graham`) → square resize → ImageNet normalise → `float32 (1,3,H,W)` |
| Inference | onnxruntime (`--onnx`, contract export `(1,5)` probs) or PyTorch ckpt (`--ckpt`: `state_dict` + `temperature` from the payload) |
| Output | 5-grade probability bars, argmax grade, soft expected grade, referable risk `P(grade ≥ 2)` vs. adjustable threshold |
| Input size | taken from the checkpoint config / ONNX metadata; `--img-size` overrides |

The PyTorch loader tolerates head-name differences between builds (shape-matched partial load and
reports how many tensors matched), so the demo keeps working as `DrNet` evolves.
