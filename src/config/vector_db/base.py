from typing import Optional

from src.config.base import BaseConfig


class BaseVectorDBConfig(BaseConfig):
    collection_name: str
    vector_dim: Optional[int] = None

