"""Search tracks.json for a bib number and list matching photo paths.

Usage:
    python scripts/search_bib.py --run results/carrera_canina_2 --bib 166

Output:
    One line per track with summary (track_id, photo count, OCR vote confidence),
    followed by the absolute path of each matching JPEG.

Exit codes:
    0 — bib found
    1 — tracks.json missing or bib not found
"""

import argparse
import json
from pathlib import Path


def run(run_dir: Path, bib: str) -> int:
    """Return 0 if hits found, 1 otherwise."""
    tracks_path = run_dir / "tracks.json"
    if not tracks_path.exists():
        print(f"ERROR: {tracks_path} not found — run 'make run-video RUN=...' first")
        return 1

    data = json.loads(tracks_path.read_text())
    hits = [t for t in data["tracks"] if t.get("bib_number") == bib]

    if not hits:
        print(f"No results for bib {bib}")
        return 1

    images_dir = run_dir / "images"
    total_photos = sum(t["photo_count"] for t in hits)
    print(f"Bib {bib}: {len(hits)} track(s), {total_photos} photo(s)\n")

    for track in hits:
        print(
            f"  track_id={track['track_id']}"
            f"  photos={track['photo_count']}"
            f"  ocr_vote={track['bib_vote_fraction']:.0%}"
            f"  votes={track['bib_votes']}"
        )
        for photo_id in track["photo_ids"]:
            print(f"    {images_dir / (photo_id + '.jpg')}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search bib number in processed tracks and list photo paths."
    )
    parser.add_argument(
        "--run", type=Path, required=True,
        help="RUN_DIR — path to the processed run (e.g. results/carrera_canina_2)",
    )
    parser.add_argument(
        "--bib", required=True,
        help="Bib number to search (string, e.g. 166)",
    )
    args = parser.parse_args()
    raise SystemExit(run(args.run, args.bib))


if __name__ == "__main__":
    main()
