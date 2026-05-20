"""Entry point: python -m src.pipeline --input <path> --output <dir>

--input can be:
  - a directory of images  →  image mode
  - a video file (.mp4, .mov, ...)  →  video mode
"""

import argparse
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import structlog

from src.config import settings
from src.detection.detector import PersonDetector
from src.faces.embedder import FaceEmbedder
from src.ingest.loader import ImageRecord, detect_input_mode, iter_video_frames, load_images
from src.linking.linker import link
from src.ocr.reader import BibReader
from src.storage.writer import ImageResult, write_image_result, write_master, write_tracks

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

log = structlog.get_logger()


@dataclass
class _TrackState:
    bib_votes: Counter = field(default_factory=Counter)
    embeddings: list[np.ndarray] = field(default_factory=list)
    photo_ids: list[str] = field(default_factory=list)


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
    track_store: dict[int, _TrackState] | None = None,
    use_tracking: bool = False,
) -> ImageResult:
    persons = detector.detect(record.image, track=use_tracking)
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
                track_id=person.track_id,
            )
        )

        if track_store is not None and person.track_id is not None:
            state = track_store.setdefault(person.track_id, _TrackState())
            if bib_text:
                state.bib_votes[bib_text] += 1
            if embedding is not None:
                state.embeddings.append(embedding)
            state.photo_ids.append(record.filename)

    result = ImageResult(
        filename=record.filename,
        width=record.width,
        height=record.height,
        processed_at=datetime.now(timezone.utc).isoformat(),
        detections=linked,
        frame_number=record.frame_number,
    )
    write_image_result(result, output_dir, image=record.image)
    return result


def _resolve_tracks(track_store: dict[int, _TrackState]) -> list[dict]:
    resolved = []
    for track_id, state in sorted(track_store.items()):
        bib_number = None
        bib_vote_fraction = 0.0
        if state.bib_votes:
            bib_number, top_count = state.bib_votes.most_common(1)[0]
            bib_vote_fraction = round(top_count / sum(state.bib_votes.values()), 3)

        avg_embedding: list[float] | None = None
        if state.embeddings:
            avg_embedding = np.mean(state.embeddings, axis=0).tolist()

        # deduplicate photo_ids preserving order
        seen: set[str] = set()
        unique_photos = [p for p in state.photo_ids if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]

        resolved.append(
            {
                "track_id": track_id,
                "bib_number": bib_number,
                "bib_vote_fraction": bib_vote_fraction,
                "bib_votes": dict(state.bib_votes),
                "embedding": avg_embedding,
                "photo_count": len(unique_photos),
                "photo_ids": unique_photos,
            }
        )
    return resolved


def run(input_path: Path, output_dir: Path) -> None:
    mode = detect_input_mode(input_path)
    use_tracking = mode == "video"
    log.info("pipeline_start", input=str(input_path), mode=mode, output=str(output_dir), tracking=use_tracking)

    detector = PersonDetector()
    bib_reader = BibReader()
    embedder = FaceEmbedder()

    track_store: dict[int, _TrackState] = {}
    all_results: list[ImageResult] = []

    for record in _records_iter(input_path, mode):
        log.info("processing_frame", filename=record.filename, frame=record.frame_number)
        result = _process_record(
            record, detector, bib_reader, embedder, output_dir,
            track_store=track_store if use_tracking else None,
            use_tracking=use_tracking,
        )
        all_results.append(result)

    if not all_results:
        log.warning("no_records_processed", input=str(input_path))
        return

    write_master(all_results, output_dir)

    if use_tracking and track_store:
        resolved = _resolve_tracks(track_store)
        write_tracks(resolved, output_dir)
        bibs_resolved = sum(1 for t in resolved if t["bib_number"])
        log.info("tracking_done", tracks=len(resolved), bibs_resolved=bibs_resolved)

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
