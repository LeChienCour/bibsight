"""Unit tests for JSON writers (no GPU required)."""

import json
import tempfile
from pathlib import Path

from src.linking.linker import LinkedDetection
from src.storage.writer import ImageResult, write_image_result, write_master


def _make_result(filename: str) -> ImageResult:
    return ImageResult(
        filename=filename,
        width=640,
        height=480,
        processed_at="2026-05-18T00:00:00+00:00",
        detections=[
            LinkedDetection(
                detection_id="test-uuid",
                person_bbox=[0, 0, 100, 200],
                person_conf=0.9,
                bib_number="42",
                bib_conf=0.85,
                face_bbox=[10, 10, 50, 50],
                face_embedding=[0.1, 0.2, 0.3],
            )
        ],
    )


def test_write_image_result() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _make_result("test.jpg")
        out = write_image_result(result, Path(tmpdir))
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["filename"] == "test.jpg"
        assert len(data["detections"]) == 1
        assert data["detections"][0]["bib_number"] == "42"


def test_write_master() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        results = [_make_result("a.jpg"), _make_result("b.jpg")]
        out = write_master(results, Path(tmpdir))
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["total_images"] == 2
        assert data["total_detections"] == 2
