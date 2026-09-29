.PHONY: setup lint format test smoke train eval export-onnx bench demo clean fetch-data probe-data distill quantize

setup:
	python3 -m pip install -e ".[dev]"

lint:
	ruff check src tests scripts demo

format:
	ruff format src tests scripts demo

test:
	pytest -q

# End-to-end pipeline proof on synthetic data: train -> eval -> export -> benchmark
smoke:
	python3 -m retinaedge.train.trainer --config configs/train/smoke.yaml
	python3 -m retinaedge.eval.evaluate --config configs/train/smoke.yaml --ckpt artifacts/smoke/best.pt --split val
	python3 -m retinaedge.export.export_onnx --config configs/train/smoke.yaml --ckpt artifacts/smoke/best.pt --out artifacts/smoke/model.onnx --img-size 64
	python3 -m retinaedge.export.benchmark --model artifacts/smoke/model.onnx --backend onnxruntime --img-size 64 --runs 30

train:
	python3 -m retinaedge.train.trainer --config $(CFG)

eval:
	python3 -m retinaedge.eval.evaluate --config $(CFG) --ckpt $(CKPT) --split val

export-onnx:
	python3 -m retinaedge.export.export_onnx --config $(CFG) --ckpt $(CKPT) --out $(OUT) --img-size 224

bench:
	python3 -m retinaedge.export.benchmark --model $(MODEL) --backend $(BACKEND) --img-size 224 --runs 50

demo:
	python3 demo/gradio_app.py

clean:
	rm -rf artifacts .pytest_cache .ruff_cache build dist *.egg-info

# --- Accuracy-loophole stack & continuous improvement -------------------------
# Escalation ladder toward the accuracy goal (default 97% referable-DR acc).
CFG ?= configs/train/online_pilot.yaml
BUDGET ?= small
TARGET ?= 0.97

improve:
	python3 -m retinaedge.train.auto_improve --config $(CFG) --budget $(BUDGET) --target $(TARGET) \
	  --state runs/improve_state.json --log runs/improvement_log.md

improve-plan:
	python3 -m retinaedge.train.auto_improve --config $(CFG) --budget $(BUDGET) --target $(TARGET) --dry-run

thresholds:
	python3 -m retinaedge.eval.threshold_search --config $(CFG) --ckpt $(CKPT) --split val --out $(OUT)

soup:
	python3 -c "from retinaedge.models.soup import make_soup; make_soup('$(INGREDIENTS)'.split(','), '$(OUT)')"

pseudo:
	python3 -m retinaedge.train.pseudo_label --config $(CFG) --ckpt $(CKPT) \
	  --images-dir $(IMAGES_DIR) --out $(OUT) --tau $(TAU)

# KGAT-token Kaggle fetch (bearer auth; see configs/data/kaggle_dr_catalog.yaml)
fetch-data:
	python3 -m retinaedge.data.kaggle_fetch --handle $(HANDLE) --out data/$(NAME)

probe-data:
	KAGGLE_API_TOKEN=$${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN} \
	python3 -m retinaedge.data.kaggle_fetch --handle $(HANDLE) --out /tmp/probe --max-bytes 1048576

# Knowledge distillation: teacher checkpoint -> mobile student backbone
distill:
	python3 -m retinaedge.train.distill --config $(CFG) --teacher-ckpt $(CKPT) \
	  --student-backbone $(BACKBONE) --out-dir artifacts/distill

# ONNX int8 dynamic quantization + mobile-fit verdict
quantize:
	python3 -m retinaedge.export.quantize_onnx --onnx $(MODEL) --out $(OUT) --img-size $(SIZE)
