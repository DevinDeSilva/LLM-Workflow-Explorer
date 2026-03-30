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
    vector_dim: int | None = 1024
    metric_type: str = "COSINE"
    index_type: str = "AUTOINDEX"
    index_params: Dict[str, Any] = Field(default_factory=dict)
    search_params: Dict[str, Any] = Field(default_factory=dict)
    consistency_level: str = "Bounded"
    object_name_max_length: int = 512
    object_description_max_length: int = 8192
    enable_dynamic_field: bool = False
    enable_bm25: bool = True
    description_enable_analyzer: bool = True
    description_enable_match: bool = True
    description_analyzer_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "tokenizer": "standard",
            "filter": ["lowercase"],
        }
    )
    bm25_function_name: str = "object_description_bm25"
    bm25_sparse_field_name: str = "object_description_sparse"
    bm25_index_type: str = "SPARSE_INVERTED_INDEX"
    bm25_metric_type: str = "BM25"
    bm25_index_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "inverted_index_algo": "DAAT_MAXSCORE",
        }
    )
    bm25_search_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "drop_ratio_search": 0.0,
        }
    )
