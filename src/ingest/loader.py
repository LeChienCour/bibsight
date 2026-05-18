from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import structlog
from PIL import Image

log = structlog.get_logger()

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m2ts"}


@dataclass
class ImageRecord:
    filename: str
    path: Path
    image: np.ndarray  # RGB uint8
    width: int
    height: int
    frame_number: int | None = None  # None for static images; frame index for video


def detect_input_mode(input_path: Path) -> str:
    """Return 'video' or 'images' based on what input_path points to."""
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            return "video"
        raise ValueError(f"Unsupported file type: {input_path.suffix!r}")
    if input_path.is_dir():
        return "images"
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def load_images(input_dir: Path) -> list[ImageRecord]:
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)

    if not paths:
        log.warning("no_images_found", dir=str(input_dir))
        return []

    log.info("loading_images", count=len(paths), dir=str(input_dir))

    records: list[ImageRecord] = []
    for path in paths:
        try:
            pil = Image.open(path).convert("RGB")
            arr = np.array(pil)
            records.append(
                ImageRecord(
                    filename=path.name,
                    path=path,
                    image=arr,
                    width=pil.width,
                    height=pil.height,
                )
            )
        except Exception:
            log.exception("failed_to_load_image", path=str(path))

    return records


def iter_video_frames(video_path: Path, frame_step: int = 5) -> Iterator[ImageRecord]:
    """Yield one ImageRecord per sampled frame (every frame_step frames)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log.info(
        "video_opened",
        path=str(video_path),
        total_frames=total_frames,
        fps=fps,
        frame_step=frame_step,
    )

    stem = video_path.stem
    frame_idx = 0

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield ImageRecord(
                    filename=f"{stem}_frame_{frame_idx:06d}",
                    path=video_path,
                    image=rgb,
                    width=width,
                    height=height,
                    frame_number=frame_idx,
                )

            frame_idx += 1
    finally:
        cap.release()
        log.info("video_closed", path=str(video_path), frames_read=frame_idx)
