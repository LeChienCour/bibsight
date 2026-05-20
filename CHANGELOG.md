# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- ByteTrack multi-frame tracking via `model.track()` — assigns consistent `track_id` per runner across frames
- Majority vote bib resolution — after processing all frames, picks the most-frequent OCR reading per `track_id`
- `results/tracks.json` — authoritative per-runner output: resolved bib, averaged face embedding, photo list
- JPEG frame saving — `results/images/*.jpg` written at 95% quality alongside per-frame JSON metadata
- `track_id` field propagated through `PersonDetection`, `LinkedDetection`, and JSON outputs
- `write_tracks()` in storage writer
- Visualizer: bib number rendered with solid black background label (`#132`) for readability
- Visualizer: confidence percentage shown above bib label
- Visualizer: top-left HUD showing all bib numbers visible in current frame

### Changed
- `detector.detect()` accepts `track=True` to switch between inference and tracking modes
- `write_image_result()` accepts optional `image: np.ndarray` to save the JPEG alongside JSON
- `link()` and `LinkedDetection` now carry `track_id`
- Pipeline passes `image` to writer; tracking only enabled in video mode

---

## [0.3.0] — 2026-05-19

### Added
- Heuristic bib crop: torso zone (30–70% of person height) fed to OCR instead of full person crop
- Visualization script (`scripts/visualize.py`): overlays bounding boxes on original video
  - Green box + bib number when bib detected; yellow box when not
  - Blue box for face detection
  - 2-second persistence of last detections between analyzed frames
- `make visualize` and `make clean` Makefile targets
- PR template (`.github/pull_request_template.md`)
- Business pipeline document (`docs/architecture.md`): processing flow, purchase flow, revenue estimates, legal notes

### Fixed
- Makefile `VIDEO` path corrected from `data/video/` to `data/videos/`

---

## [0.2.0] — 2026-05-18

### Changed
- Switched OCR from PaddleOCR to EasyOCR (PyTorch-based, stable on WSL2 GPU)
- Pinned `numpy>=2.0` and `opencv-python-headless>=4.10` to resolve ABI conflict
- Fixed `pyproject.toml` build backend (`setuptools.build_meta`)

### Added
- `structlog` JSON logging throughout pipeline
- `video_frame_step` config setting (`RPT_VIDEO_FRAME_STEP`, default 5)

---

## [0.1.0] — 2026-05-17

### Added
- Initial project scaffold: `src/ingest`, `src/detection`, `src/ocr`, `src/faces`, `src/linking`, `src/storage`
- YOLOv8 person detection (`PersonDetector`)
- PaddleOCR bib reader (`BibReader`) — replaced in 0.2.0
- InsightFace face embedder (`FaceEmbedder`, ArcFace `buffalo_l` 512-d)
- Pipeline orchestrator (`src/pipeline.py`) — image and video modes
- `pydantic-settings` config with `RPT_` prefix
- `Makefile` with `install`, `run`, `run-video`, `format`, `lint`, `test` targets
