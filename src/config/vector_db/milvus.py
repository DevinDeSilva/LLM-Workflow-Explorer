import os
from typing import Any, Dict

from pydantic import Field

from src.config.vector_db.base import BaseVectorDBConfig


class MilvusVectorDBConfig(BaseVectorDBConfig):
    uri: str = Field(
        default_factory=lambda: os.getenv("MILVUS_URI", "http://localhost:19530")
    )
    token: str = Field(
        default_factory=lambda: os.getenv("MILVUS_TOKEN", "root:Milvus")
    )
    db_name: str = Field(
        default_factory=lambda: os.getenv("MILVUS_DB_NAME", "default")
    )
    collection_name: str = "object_store"
    vector_dim: int | None = None
    metric_type: str = "COSINE"
    index_type: str = "AUTOINDEX"
    index_params: Dict[str, Any] = Field(default_factory=dict)
    search_params: Dict[str, Any] = Field(default_factory=dict)
    consistency_level: str = "Bounded"
    object_name_max_length: int = 512
    object_description_max_length: int = 8192
    enable_dynamic_field: bool = False
