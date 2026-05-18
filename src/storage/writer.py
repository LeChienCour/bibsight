import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import structlog

from src.linking.linker import LinkedDetection

log = structlog.get_logger()


@dataclass
class ImageResult:
    filename: str
    width: int
    height: int
    processed_at: str
    detections: list[LinkedDetection]
    frame_number: int | None = None  # populated for video frames


def _serialize(result: ImageResult) -> dict:
    d = asdict(result)
    # face_embedding is list[float] already via linker.link()
    return d


def write_image_result(result: ImageResult, output_dir: Path) -> Path:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(result.filename).stem
    out_path = images_dir / f"{stem}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_serialize(result), f, indent=2, ensure_ascii=False)

    log.info("wrote_image_result", path=str(out_path), detections=len(result.detections))
    return out_path


def write_master(results: list[ImageResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "batch_results.json"

    total_detections = sum(len(r.detections) for r in results)
    master = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "total_images": len(results),
        "total_detections": total_detections,
        "images": [_serialize(r) for r in results],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)

    log.info("wrote_master", path=str(out_path), images=len(results), detections=total_detections)
    return out_path
