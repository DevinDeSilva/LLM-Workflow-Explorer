from icecream import ic
from typing import Optional

from src.vector_db.base import BaseVectorDB


class VectorDB:
    def __new__(cls, vector_db_type: str, **kwargs):
        return cls.__create_concrete__(vector_db_type, **kwargs)

    @classmethod
    def __create_concrete__(
        cls,
        vector_db_type: str,
        **kwargs,
    ) -> BaseVectorDB:
        if vector_db_type == "milvus":
            from src.config.vector_db.milvus import MilvusVectorDBConfig
            from src.vector_db.milvus import MilvusVectorDB

            config = MilvusVectorDBConfig(**kwargs)
            ic(config)
            return MilvusVectorDB(config)

        raise NotImplementedError
