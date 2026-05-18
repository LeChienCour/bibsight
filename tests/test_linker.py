"""Unit tests for linking logic (no GPU required)."""

import numpy as np

from src.linking.linker import link


def test_link_with_all_fields() -> None:
    emb = np.ones(512, dtype=np.float32)
    det = link(
        person_bbox=[0, 0, 100, 200],
        person_conf=0.9,
        bib_number="42",
        bib_conf=0.85,
        embedding=emb,
        face_bbox=[10, 10, 50, 50],
    )
    assert det.bib_number == "42"
    assert det.person_conf == 0.9
    assert len(det.face_embedding) == 512
    assert det.detection_id  # non-empty UUID


def test_link_without_face_or_bib() -> None:
    det = link(
        person_bbox=[0, 0, 100, 200],
        person_conf=0.7,
        bib_number=None,
        bib_conf=0.0,
        embedding=None,
        face_bbox=None,
    )
    assert det.bib_number is None
    assert det.face_embedding is None
    assert det.face_bbox is None
