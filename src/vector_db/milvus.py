from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.config.vector_db.milvus import MilvusVectorDBConfig
from src.vector_db.base import BaseVectorDB

try:
    from pymilvus import DataType, Function, FunctionType, MilvusClient
except (ImportError, ModuleNotFoundError):
    DataType = None
    Function = None
    FunctionType = None
    MilvusClient = None


class MilvusVectorDB(BaseVectorDB):
    def __init__(self, config: MilvusVectorDBConfig) -> None:
        super().__init__(config)

    def _create_client(self) -> Any:
        if MilvusClient is None:
            raise ModuleNotFoundError("Please install pymilvus to use the Milvus vector DB client.")

        kwargs: Dict[str, Any] = {"uri": self.config.uri}
        if self.config.token:
            kwargs["token"] = self.config.token
        if self.config.db_name:
            kwargs["db_name"] = self.config.db_name

        return MilvusClient(**kwargs)

    @property
    def collection_name(self) -> str:
        return self.config.collection_name

    def build_db(self, overwrite: bool = False) -> str:
        vector_dim = self.config.vector_dim
        if not vector_dim:
            raise ValueError("`vector_dim` must be provided before the collection can be created.")
        if self.config.enable_bm25 and (Function is None or FunctionType is None):
            raise ModuleNotFoundError(
                "BM25 full-text search requires a pymilvus version with Function and FunctionType support."
            )

        if self._collection_exists():
            if not overwrite:
                return self.collection_name
            self.client.drop_collection(collection_name=self.collection_name)

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=self.config.enable_dynamic_field,
        )
        schema.add_field(
            field_name="object_name",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=self.config.object_name_max_length,
        )
        schema.add_field(
            field_name="object_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=vector_dim,
        )
        schema.add_field(
            field_name="metadata",
            datatype=DataType.JSON,
        )
        object_description_field: Dict[str, Any] = {
            "field_name": "object_description",
            "datatype": DataType.VARCHAR,
            "max_length": self.config.object_description_max_length,
        }
        if self.config.enable_bm25:
            object_description_field["enable_analyzer"] = self.config.description_enable_analyzer
            object_description_field["enable_match"] = self.config.description_enable_match
            object_description_field["analyzer_params"] = self.config.description_analyzer_params
        schema.add_field(**object_description_field)

        if self.config.enable_bm25:
            schema.add_field(
                field_name=self.config.bm25_sparse_field_name,
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            schema.add_function(
                Function(
                    name=self.config.bm25_function_name,
                    function_type=FunctionType.BM25,
                    input_field_names=["object_description"],
                    output_field_names=[self.config.bm25_sparse_field_name],
                )
            )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="object_vector",
            index_type=self.config.index_type,
            metric_type=self.config.metric_type,
            params=self.config.index_params,
        )
        if self.config.enable_bm25:
            index_params.add_index(
                field_name=self.config.bm25_sparse_field_name,
                index_type=self.config.bm25_index_type,
                metric_type=self.config.bm25_metric_type,
                params=self.config.bm25_index_params,
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level=self.config.consistency_level,
        )
        return self.collection_name

    def insert(
        self,
        object_name: Optional[str] = None,
        object_vector: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        object_description: str = "",
        records: Optional[List[Dict[str, Any]]] = None,
        build_if_missing: bool = True,
    ) -> List[Any]:
        normalized_records = self._normalize_records(
            object_name=object_name,
            object_vector=object_vector,
            metadata=metadata,
            object_description=object_description,
            records=records,
        )

        if not normalized_records:
            raise ValueError("At least one record must be provided for insert.")

        if self.config.vector_dim is None:
            self.config.vector_dim = len(normalized_records[0]["object_vector"])

        self._validate_vectors(normalized_records)

        if build_if_missing and not self._collection_exists():
            self.build_db()
        elif not build_if_missing and not self._collection_exists():
            raise ValueError(f"Collection `{self.collection_name}` does not exist. Run `build_db()` first.")

        return self.client.insert(
            collection_name=self.collection_name,
            data=normalized_records,
        )

    def search(
        self,
        query_vector: Sequence[float],
        limit: int = 10,
        filter: str = "",
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._collection_exists():
            raise ValueError(f"Collection `{self.collection_name}` does not exist. Run `build_db()` first.")

        query = self._normalize_vector(query_vector)
        self._validate_vector_dim(query)

        resolved_output_fields = output_fields or [
            "object_name",
            "metadata",
            "object_description",
        ]
        resolved_search_params = search_params or {
            "metric_type": self.config.metric_type,
            "params": self.config.search_params,
        }

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field="object_vector",
            filter=filter or None,
            limit=limit,
            output_fields=resolved_output_fields,
            search_params=resolved_search_params,
        )

        return self._format_search_results(results[0] if results else [])

    def bm25_search(
        self,
        query_text: str,
        limit: int = 10,
        filter: str = "",
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.config.enable_bm25:
            raise ValueError("BM25 full-text search is disabled for this collection.")
        if not self._collection_exists():
            raise ValueError(f"Collection `{self.collection_name}` does not exist. Run `build_db()` first.")

        normalized_query = query_text.strip()
        if not normalized_query:
            raise ValueError("`query_text` must not be empty.")

        resolved_output_fields = output_fields or [
            "object_name",
            "metadata",
            "object_description",
        ]
        resolved_search_params = search_params or {
            "metric_type": self.config.bm25_metric_type,
            "params": self.config.bm25_search_params,
        }

        results = self.client.search(
            collection_name=self.collection_name,
            data=[normalized_query],
            anns_field=self.config.bm25_sparse_field_name,
            filter=filter or None,
            limit=limit,
            output_fields=resolved_output_fields,
            search_params=resolved_search_params,
        )

        return self._format_search_results(results[0] if results else [])

    def _collection_exists(self) -> bool:
        return self.collection_name in self.client.list_collections()

    def _normalize_records(
        self,
        object_name: Optional[str],
        object_vector: Optional[Sequence[float]],
        metadata: Optional[Dict[str, Any]],
        object_description: str,
        records: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if records is not None:
            return [
                {
                    "object_name": record["object_name"],
                    "object_vector": self._normalize_vector(record["object_vector"]),
                    "metadata": record.get("metadata", {}),
                    "object_description": record.get("object_description", ""),
                }
                for record in records
            ]

        if object_name is None or object_vector is None:
            raise ValueError(
                "`object_name` and `object_vector` are required when `records` is not provided."
            )

        return [
            {
                "object_name": object_name,
                "object_vector": self._normalize_vector(object_vector),
                "metadata": metadata or {},
                "object_description": object_description,
            }
        ]

    def _normalize_vector(self, vector: Sequence[float]) -> List[float]:
        if len(vector) == 0:
            raise ValueError("Vectors must not be empty.")

        return [float(value) for value in vector]

    def _validate_vectors(self, records: List[Dict[str, Any]]) -> None:
        for record in records:
            self._validate_vector_dim(record["object_vector"])

    def _validate_vector_dim(self, vector: Sequence[float]) -> None:
        expected_dim = self.config.vector_dim
        if expected_dim is None:
            return

        if len(vector) != expected_dim:
            raise ValueError(
                f"Expected vectors with dimension {expected_dim}, received {len(vector)}."
            )

    def _format_search_results(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted_results: List[Dict[str, Any]] = []

        for hit in hits:
            entity = hit.get("entity", {})
            result: Dict[str, Any] = {
                "score": hit.get("distance", hit.get("score")),
            }

            if "id" in hit and "object_name" not in entity:
                result["object_name"] = hit["id"]

            result.update(entity)

            if "object_name" not in result and "object_name" in hit:
                result["object_name"] = hit["object_name"]
            if "metadata" not in result and "metadata" in hit:
                result["metadata"] = hit["metadata"]
            if "object_description" not in result and "object_description" in hit:
                result["object_description"] = hit["object_description"]

            formatted_results.append(result)

        return formatted_results
