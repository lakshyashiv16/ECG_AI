VENV = venv/bin/activate

.DEFAULT_GOAL := dashboard

dashboard:
	source $(VENV) && python dashboard/app.py

train:
	source $(VENV) && python src/train.py --model cnn1d --dataset mitbih --epochs 30 --batch_size 64 --no_mixed_precision

smoke:
	source $(VENV) && python src/train.py --model cnn1d --dataset mitbih --epochs 2 --subset 100 --no_mixed_precision

test:
	source $(VENV) && python -m pytest tests/ -v

install:
	source $(VENV) && pip install -r requirements.txt

.PHONY: dashboard train smoke test install
