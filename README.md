# Bibsight

Local GPU pipeline for sports photography: detects runners, reads bib numbers via OCR, and extracts face embeddings from drone/camera video or photo batches. Designed to power a photo-sales platform where athletes find and purchase their race photos by bib number or selfie.

## How it works

```
video / photos
      ↓
YOLO — person detection
      ↓
ByteTrack — multi-frame tracking (track_id per runner)
      ↓
EasyOCR — bib number read from torso crop (30–70% height)
      ↓
InsightFace — face embedding (ArcFace 512-d)
      ↓
Majority vote — resolve final bib per track_id
      ↓
results/tracks.json  ←  {track_id, bib_number, embedding, photo_ids}
results/images/*.jpg ←  extracted frames (originals)
```

## Requirements

- Python 3.11+
- CUDA-capable GPU (tested on RTX 4070, WSL2)
- [uv](https://github.com/astral-sh/uv) or pip

## Setup

```bash
pip install -e ".[dev]"
```

Place YOLO weights in the repo root (or set `RPT_YOLO_MODEL`):

```bash
# download yolov8x (recommended)
python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"
```

## Usage

```bash
# process a video
make run-video VIDEO=data/videos/race.mp4

# process a folder of images
make run INPUT=data/images/

# render bounding boxes onto video
make visualize VIDEO=data/videos/race.mp4

# clean outputs
make clean
```

### Environment variables (prefix `RPT_`)

| Variable | Default | Description |
|---|---|---|
| `RPT_YOLO_MODEL` | `yolov8n.pt` | YOLO weights file |
| `RPT_DEVICE` | `cuda` | `cuda` or `cpu` |
| `RPT_VIDEO_FRAME_STEP` | `5` | Sample 1 of every N frames |
| `RPT_YOLO_CONF_PERSON` | `0.5` | Person detection confidence threshold |
| `RPT_OCR_MIN_CONF` | `0.7` | Minimum OCR confidence to accept a bib reading |

Create a `.env` file in the repo root to override defaults.

## Output

| File | Description |
|---|---|
| `results/tracks.json` | Authoritative per-runner table: bib, embedding, photo list |
| `results/batch_results.json` | Per-frame detections (raw, before majority vote) |
| `results/images/*.jpg` | Extracted frames at 95% JPEG quality |
| `results/images/*.json` | Per-frame metadata |
| `results/visualized.mp4` | Debug video with bounding boxes overlaid |

### `tracks.json` schema

```json
{
  "tracks": [
    {
      "track_id": 7,
      "bib_number": "132",
      "bib_vote_fraction": 0.87,
      "bib_votes": {"132": 14, "13": 2},
      "embedding": [0.123, ...],
      "photo_count": 18,
      "photo_ids": ["frame_000010", "frame_000015", ...]
    }
  ]
}
```

## Development

```bash
make format   # ruff + black
make lint     # check only
make test     # pytest + coverage
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full business pipeline (S3, DynamoDB, OpenSearch, Lambda, payments).

## Stack

| Component | Library |
|---|---|
| Person detection | Ultralytics YOLOv8 |
| Multi-frame tracking | ByteTrack (via Ultralytics) |
| Bib OCR | EasyOCR |
| Face embedding | InsightFace `buffalo_l` (ArcFace 512-d) |
| Config | pydantic-settings |
| Logging | structlog (JSON) |
