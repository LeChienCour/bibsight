import numpy as np
import structlog
from insightface.app import FaceAnalysis

from src.config import settings

log = structlog.get_logger()


class FaceEmbedder:
    def __init__(self) -> None:
        ctx_id = settings.insightface_ctx_id if settings.device == "cuda" else -1
        self._app = FaceAnalysis(
            name=settings.insightface_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        log.info("insightface_loaded", model=settings.insightface_model, ctx_id=ctx_id)

    def embed(self, crop: np.ndarray) -> tuple[np.ndarray | None, list[int] | None]:
        """Return (embedding[512], face_bbox) for the highest-confidence face in crop.

        Returns (None, None) if no face detected.
        """
        faces = self._app.get(crop)

        if not faces:
            return None, None

        # Pick face with largest det_score
        face = max(faces, key=lambda f: f.det_score)
        bbox = [int(v) for v in face.bbox.tolist()]
        return face.embedding, bbox
