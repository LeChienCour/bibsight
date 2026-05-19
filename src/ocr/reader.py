import re

import easyocr
import numpy as np
import structlog

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
        self._ocr = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        log.info("easyocr_loaded", gpu=use_gpu)

    def read(self, crop: np.ndarray) -> tuple[str | None, float]:
        results = self._ocr.readtext(crop, detail=1)

        candidates: list[tuple[str, float]] = []
        for _, text, conf in results:
            if conf < settings.ocr_min_conf:
                continue
            bib = _extract_bib(text)
            if bib:
                candidates.append((bib, conf))

        if not candidates:
            return None, 0.0

        best = max(candidates, key=lambda x: (len(x[0]), x[1]))
        return best
