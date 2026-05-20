import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinkedDetection:
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    person_bbox: list[int] = field(default_factory=list)
    person_conf: float = 0.0
    track_id: int | None = None
    bib_number: str | None = None
    bib_conf: float = 0.0
    face_bbox: list[int] | None = None
    face_embedding: list[float] | None = None


def link(
    person_bbox: list[int],
    person_conf: float,
    bib_number: str | None,
    bib_conf: float,
    embedding: np.ndarray | None,
    face_bbox: list[int] | None,
    track_id: int | None = None,
) -> LinkedDetection:
    return LinkedDetection(
        person_bbox=person_bbox,
        person_conf=person_conf,
        track_id=track_id,
        bib_number=bib_number,
        bib_conf=bib_conf,
        face_bbox=face_bbox,
        face_embedding=embedding.tolist() if embedding is not None else None,
    )
