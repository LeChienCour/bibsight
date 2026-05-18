import re

import numpy as np
import structlog
from paddleocr import PaddleOCR

from src.config import settings

log = structlog.get_logger()

_BIB_PATTERN = re.compile(r"^[A-Z0-9]{1,5}$")


def _extract_bib(text: str) -> str | None:
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if _BIB_PATTERN.match(cleaned):
        return cleaned
    return None


class BibReader:
    def __init__(self) -> None:
        use_gpu = settings.device == "cuda"
        self._ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=use_gpu, show_log=False)
        log.info("paddleocr_loaded", gpu=use_gpu)

    def read(self, crop: np.ndarray) -> tuple[str | None, float]:
        """Return (bib_number, confidence). bib_number is None if not found."""
        result = self._ocr.ocr(crop, cls=True)

        if not result or not result[0]:
            return None, 0.0

        candidates: list[tuple[str, float]] = []
        for line in result[0]:
            text, conf = line[1]
            if conf < settings.ocr_min_conf:
                continue
            bib = _extract_bib(text)
            if bib:
                candidates.append((bib, conf))

        if not candidates:
            return None, 0.0

        # Prefer longest bib (more digits = more specific), then highest conf
        best = max(candidates, key=lambda x: (len(x[0]), x[1]))
        return best
