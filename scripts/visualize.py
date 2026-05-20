"""Render bounding boxes from pipeline results onto video frames.

Usage:
    python scripts/visualize.py \
        --input  data/videos/my_video.mp4 \
        --results results/batch_results.json \
        --output  results/visualized.mp4
"""

import argparse
import json
from pathlib import Path

import cv2


def _put_label(frame, text: str, x: int, y: int, color: tuple, font_scale: float = 0.6, thickness: int = 2) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    top = max(y - th - baseline - 4, 0)
    cv2.rectangle(frame, (x, top), (x + tw + 4, top + th + baseline + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 2, top + th + 2), font, font_scale, color, thickness)


def _draw_frame(frame, detections: list[dict]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det["person_bbox"]
        bib = det.get("bib_number")
        conf = det.get("person_conf", 0.0)
        has_face = det.get("face_bbox") is not None

        # person box — green if bib found, yellow otherwise
        color = (0, 255, 0) if bib else (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if bib:
            # large bib number + confidence
            _put_label(frame, f"#{bib}", x1, y1, color, font_scale=0.9, thickness=2)
            _put_label(frame, f"{det.get('bib_conf', 0):.0%}", x1, y1 - 28, (200, 200, 200), font_scale=0.5, thickness=1)
        else:
            _put_label(frame, f"person {conf:.2f}", x1, y1, color, font_scale=0.55, thickness=1)

        if has_face:
            fx1, fy1, fx2, fy2 = det["face_bbox"]
            # face coords are relative to person crop — offset by person bbox
            cv2.rectangle(frame, (x1 + fx1, y1 + fy1), (x1 + fx2, y1 + fy2), (255, 100, 0), 1)


def run(input_path: Path, results_path: Path, output_path: Path) -> None:
    with results_path.open() as f:
        data = json.load(f)

    # index results by frame_number for fast lookup
    by_frame: dict[int, list[dict]] = {}
    for img in data["images"]:
        fn = img.get("frame_number")
        if fn is not None:
            by_frame[fn] = img["detections"]

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    last_detections: list[dict] = []
    last_analyzed = -999
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in by_frame:
            last_detections = by_frame[frame_idx]
            last_analyzed = frame_idx
        # persist last detections up to 2 seconds worth of frames
        active = last_detections if (frame_idx - last_analyzed) <= int(fps * 2) else []
        _draw_frame(frame, active)

        # top-left HUD: unique bibs visible in this frame
        bibs_in_frame = [d["bib_number"] for d in active if d.get("bib_number")]
        hud = f"Bibs: {', '.join(sorted(set(bibs_in_frame)))}" if bibs_in_frame else "Bibs: —"
        _put_label(frame, hud, 8, 30, (0, 255, 255), font_scale=0.65, thickness=2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Done: {output_path}  ({frame_idx} frames)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Original video")
    parser.add_argument("--results", type=Path, default=Path("results/batch_results.json"))
    parser.add_argument("--output", type=Path, default=Path("results/visualized.mp4"))
    args = parser.parse_args()
    run(args.input, args.results, args.output)


if __name__ == "__main__":
    main()
