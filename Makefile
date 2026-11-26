.PHONY: install dev test lint smoke train chat eval merge data

install:        ## Install the toolkit
	pip install -e .

dev:            ## Install with test/lint tools
	pip install -e ".[dev]"

test:           ## Run unit tests (no GPU needed)
	pytest -q

lint:
	ruff check src tests scripts

data:           ## Rebuild the sample dataset
	python scripts/build_sample_data.py

smoke:          ## 10-step run on a tiny model — works on CPU
	python -m lora_ft.train -c configs/smoke_test.yaml

train:          ## LoRA on Qwen3-0.6B (free Colab T4 is enough)
	python -m lora_ft.train -c configs/qwen3_0.6b_lora.yaml

chat:
	python -m lora_ft.chat --adapter outputs/qwen3-0.6b-lora/adapter --compare

eval:
	python -m lora_ft.evaluate --adapter outputs/qwen3-0.6b-lora/adapter --data data/recruiter_assistant/eval.jsonl

merge:
	python -m lora_ft.merge --adapter outputs/qwen3-0.6b-lora/adapter --out outputs/qwen3-0.6b-merged
