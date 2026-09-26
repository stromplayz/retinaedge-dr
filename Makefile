.PHONY: setup lint format test smoke train eval export-onnx bench demo clean

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
