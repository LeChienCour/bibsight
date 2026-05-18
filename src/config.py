from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RPT_", env_file=".env", extra="ignore")

    yolo_model: str = "yolov8n.pt"
    yolo_conf_person: float = 0.5

    insightface_model: str = "buffalo_l"
    insightface_ctx_id: int = 0

    ocr_min_conf: float = 0.7

    video_frame_step: int = 5  # process 1 of every N frames

    input_dir: Path = Path("data/images")
    output_dir: Path = Path("results")

    device: str = "cuda"


settings = Settings()
