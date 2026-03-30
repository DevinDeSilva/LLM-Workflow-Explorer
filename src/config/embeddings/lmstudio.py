import os
from pydantic import Field

from src.config.embeddings.base import BaseEmbeddingsConfig


class LMStudioEmbeddingsConfig(BaseEmbeddingsConfig):
    model: str = "text-embedding-bge-large-en-v1.5"
    dimensions: int | None = None
    base_url: str = Field(
        default_factory=lambda: os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
    )
