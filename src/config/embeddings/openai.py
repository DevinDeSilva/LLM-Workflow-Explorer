import os
from pydantic import Field

from src.config.embeddings.base import BaseEmbeddingsConfig


class OpenAIEmbeddingsConfig(BaseEmbeddingsConfig):
    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )  # type: ignore
    model: str = "text-embedding-3-large"
    dimensions: int | None = None
