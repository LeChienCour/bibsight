from dataclasses import dataclass

import numpy as np
import structlog
from ultralytics import YOLO

from src.config import settings

log = structlog.get_logger()


@dataclass
class PersonDetection:
    bbox: list[int]  # [x1, y1, x2, y2]
    conf: float


class PersonDetector:
    def __init__(self) -> None:
        self._model = YOLO(settings.yolo_model)
        log.info("yolo_loaded", model=settings.yolo_model, device=settings.device)

    def detect(self, image: np.ndarray) -> list[PersonDetection]:
        results = self._model(
            image,
            conf=settings.yolo_conf_person,
            classes=[0],  # COCO class 0 = person
            device=settings.device,
            verbose=False,
        )

        detections: list[PersonDetection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    PersonDetection(
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        conf=float(box.conf[0]),
                    )
                )

        return detections
