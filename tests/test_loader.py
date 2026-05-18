"""Unit tests for loader routing logic (no GPU, no real video required)."""

from pathlib import Path

import pytest

from src.ingest.loader import detect_input_mode


def test_detect_mode_image_dir(tmp_path: Path) -> None:
    assert detect_input_mode(tmp_path) == "images"


def test_detect_mode_video_file(tmp_path: Path) -> None:
    video = tmp_path / "race.mp4"
    video.touch()
    assert detect_input_mode(video) == "video"


@pytest.mark.parametrize("ext", [".mov", ".avi", ".mkv", ".mts", ".m2ts"])
def test_detect_mode_video_extensions(tmp_path: Path, ext: str) -> None:
    video = tmp_path / f"clip{ext}"
    video.touch()
    assert detect_input_mode(video) == "video"


def test_detect_mode_unsupported_file(tmp_path: Path) -> None:
    bad = tmp_path / "data.csv"
    bad.touch()
    with pytest.raises(ValueError, match="Unsupported file type"):
        detect_input_mode(bad)


def test_detect_mode_missing_path() -> None:
    with pytest.raises(FileNotFoundError):
        detect_input_mode(Path("/nonexistent/path"))
