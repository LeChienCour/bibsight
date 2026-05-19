.PHONY: install format lint run run-video visualize test clean

INPUT  ?= data/images
VIDEO  ?= data/videos/race.mp4
OUTPUT ?= results

install:
	pip install -e ".[dev]"

format:
	ruff check --fix src/ tests/ scripts/
	black src/ tests/ scripts/

lint:
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/

run:
	python -m src.pipeline --input $(INPUT) --output $(OUTPUT)

run-video:
	python -m src.pipeline --input $(VIDEO) --output $(OUTPUT)

visualize:
	python scripts/visualize.py \
		--input $(VIDEO) \
		--results $(OUTPUT)/batch_results.json \
		--output $(OUTPUT)/visualized.mp4

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	rm -rf results/images results/batch_results.json results/visualized.mp4
