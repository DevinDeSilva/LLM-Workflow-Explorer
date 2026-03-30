from src.config.base import BaseConfig


class BaseVectorDBConfig(BaseConfig):
    collection_name: str
    vector_dim: int | None = None
    enable_bm25: bool = True
