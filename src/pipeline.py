"""Entry point: python -m src.pipeline --input <path> --output <dir>

--input can be:
  - a directory of images  →  image mode
  - a video file (.mp4, .mov, ...)  →  video mode
"""

import argparse
import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import structlog

from src.config import settings
from src.detection.detector import PersonDetector
from src.faces.embedder import FaceEmbedder
from src.ingest.loader import ImageRecord, detect_input_mode, iter_video_frames, load_images
from src.linking.linker import link
from src.ocr.reader import BibReader
from src.storage.writer import ImageResult, write_image_result, write_master

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

log = structlog.get_logger()


def _records_iter(input_path: Path, mode: str) -> Iterator[ImageRecord]:
    if mode == "video":
        yield from iter_video_frames(input_path, frame_step=settings.video_frame_step)
    else:
        yield from load_images(input_path)


def _process_record(
    record: ImageRecord,
    detector: PersonDetector,
    bib_reader: BibReader,
    embedder: FaceEmbedder,
    output_dir: Path,
) -> ImageResult:
    persons = detector.detect(record.image)
    linked = []

    for person in persons:
        x1, y1, x2, y2 = person.bbox
        crop = record.image[y1:y2, x1:x2]

        # Heuristic bib zone: torso area (30–70% height, full width)
        h = y2 - y1
        bib_y1 = int(h * 0.30)
        bib_y2 = int(h * 0.70)
        bib_crop = crop[bib_y1:bib_y2, :]

        bib_text, bib_conf = bib_reader.read(bib_crop)
        embedding, face_bbox = embedder.embed(crop)

        linked.append(
            link(
                person_bbox=person.bbox,
                person_conf=person.conf,
                bib_number=bib_text,
                bib_conf=bib_conf,
                embedding=embedding,
                face_bbox=face_bbox,
            )
        )

    result = ImageResult(
        filename=record.filename,
        width=record.width,
        height=record.height,
        processed_at=datetime.now(timezone.utc).isoformat(),
        detections=linked,
        frame_number=record.frame_number,
    )
    write_image_result(result, output_dir)
    return result


def run(input_path: Path, output_dir: Path) -> None:
    mode = detect_input_mode(input_path)
    log.info("pipeline_start", input=str(input_path), mode=mode, output=str(output_dir))

    detector = PersonDetector()
    bib_reader = BibReader()
    embedder = FaceEmbedder()

    all_results: list[ImageResult] = []

    for record in _records_iter(input_path, mode):
        log.info("processing_frame", filename=record.filename, frame=record.frame_number)
        result = _process_record(record, detector, bib_reader, embedder, output_dir)
        all_results.append(result)

    if not all_results:
        log.warning("no_records_processed", input=str(input_path))
        return

    write_master(all_results, output_dir)
    log.info("pipeline_done", mode=mode, processed=len(all_results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bibsight — bib & face detection pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Image directory or video file (.mp4, .mov, ...)",
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
