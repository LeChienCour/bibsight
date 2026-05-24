.PHONY: install format lint run run-video visualize search test clean help

# ── Variables ─────────────────────────────────────────────────────────────────
# image directory (run target)
INPUT   ?= data/images
# video file (run-video / visualize targets)
VIDEO   ?= data/videos/race.mp4
# base output directory
OUTPUT  ?= results
# run name → outputs land in OUTPUT/RUN
RUN     ?= default
# bib number to search (search target)
BIB     ?=

# e.g. results/carrera_canina
RUN_DIR  = $(OUTPUT)/$(RUN)
RESULTS  = $(RUN_DIR)/batch_results.json
VIS_OUT  = $(RUN_DIR)/visualized.mp4

help:                           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────────────────
install:                        ## Install package + dev deps (editable)
	pip install -e ".[dev]"

# ── Code quality ──────────────────────────────────────────────────────────────
format:                         ## Auto-fix lint + format
	ruff check --fix src/ tests/ scripts/
	black src/ tests/ scripts/

lint:                           ## Check lint + format (no writes)
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/

# ── Pipeline ──────────────────────────────────────────────────────────────────
run:                            ## Process image directory  → INPUT, RUN
	python3 -m src.pipeline --input $(INPUT) --output $(RUN_DIR)

run-video:                      ## Process video file       → VIDEO, RUN
	python3 -m src.pipeline --input $(VIDEO) --output $(RUN_DIR)

visualize:                      ## Render detections onto video → VIDEO, RUN, VIS_OUT
	python3 scripts/visualize.py \
		--input $(VIDEO) \
		--results $(RESULTS) \
		--output $(VIS_OUT)

search:                         ## List photos for a bib number → RUN, BIB
	python3 scripts/search_bib.py --run $(RUN_DIR) --bib $(BIB)

# ── Tests ─────────────────────────────────────────────────────────────────────
test:                           ## Run test suite with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:                          ## Delete all outputs for a run → RUN
	rm -rf $(RUN_DIR)
