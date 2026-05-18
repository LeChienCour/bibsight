.PHONY: install format lint run run-video test

INPUT  ?= data/images
VIDEO  ?= data/video/race.mp4
OUTPUT ?= results

install:
	pip install -e ".[dev]"

format:
	ruff check --fix src/ tests/
	black src/ tests/

lint:
	ruff check src/ tests/
	black --check src/ tests/

run:
	python -m src.pipeline --input $(INPUT) --output $(OUTPUT)

run-video:
	python -m src.pipeline --input $(VIDEO) --output $(OUTPUT)

test:
	pytest tests/ -v --cov=src --cov-report=term-missing
