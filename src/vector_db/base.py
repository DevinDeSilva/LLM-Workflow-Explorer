from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from src.config.vector_db.base import BaseVectorDBConfig


class BaseVectorDB(ABC):
    def __init__(self, config: "BaseVectorDBConfig"):
        self.config = config
        self.client = self._create_client()

    @abstractmethod
    def _create_client(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build_db(self, overwrite: bool = False) -> str:
        raise NotImplementedError

    @abstractmethod
    def insert(
        self,
        object_name: Optional[str] = None,
        object_vector: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        object_description: str = "",
        records: Optional[List[Dict[str, Any]]] = None,
        build_if_missing: bool = True,
    ) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        limit: int = 10,
        filter: str = "",
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def bm25_search(
        self,
        query_text: str,
        limit: int = 10,
        filter: str = "",
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError
