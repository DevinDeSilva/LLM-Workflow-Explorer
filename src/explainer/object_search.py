from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from rdflib.namespace import RDF

from src.config.object_search import ObjectSearchConfig
from src.embeddings.base import BaseEmbeddings
from src.utils.graph_manager import GraphManager
from src.vector_db.base import BaseVectorDB


class ObjectSearch:
    def __init__(
        self,
        graph_manager: GraphManager,
        embeddings_model: BaseEmbeddings,
        vector_db: BaseVectorDB,
        config: ObjectSearchConfig
    ) -> None:
        self.graph_manager = graph_manager
        self.config = config
        self.embeddings_model = embeddings_model
        self.vector_db = vector_db
        
        self.index_objects()

    def get_all_objects(self) -> List[str]:
        query_df = self.graph_manager.query(self.config.all_objects_query)
        if query_df.empty or "object" not in query_df.columns:
            return []

        return list(dict.fromkeys(query_df["object"].tolist()))

    def get_object_properties(self, object_uri: str) -> List[Dict[str, str]]:
        query_df = self.graph_manager.query(
            self._render_query(
                self.config.object_properties_query,
                object_uri=object_uri,
            )
        )
        if query_df.empty:
            return []

        return query_df.to_dict("records")

    def build_object_description(self, object_uri: str) -> str:
        object_name = self._format_term(object_uri)
        properties = self.get_object_properties(object_uri)

        description_parts = [f"Object: {object_name}."]
        if not properties:
            return " ".join(description_parts)

        object_types: List[str] = []
        property_lines: List[str] = []

        for property_row in sorted(
            properties,
            key=lambda row: (row["predicate"], row["value"]),
        ):
            predicate = property_row["predicate"]
            value = property_row["value"]
            formatted_value = self._format_term(value)

            if predicate == str(RDF.type):
                object_types.append(formatted_value)
                continue

            property_lines.append(f"{self._format_term(predicate)}: {formatted_value}")

        if object_types:
            description_parts.append(
                "Types: " + ", ".join(sorted(dict.fromkeys(object_types))) + "."
            )

        if property_lines:
            description_parts.append(
                "Properties: " + "; ".join(property_lines) + "."
            )

        return " ".join(description_parts)

    def build_index_records(self) -> List[Dict[str, Any]]:
        object_uris = self.get_all_objects()
        if not object_uris:
            return []

        object_names = [self._format_term(object_uri) for object_uri in object_uris]
        descriptions = [
            self.build_object_description(object_uri)
            for object_uri in object_uris
        ]
        vectors = self.embeddings_model.embed_documents(descriptions)

        return [
            {
                "object_name": object_name,
                "object_vector": object_vector,
                "metadata": {},
                "object_description": object_description,
            }
            for object_name, object_vector, object_description in zip(
                object_names,
                vectors,
                descriptions,
            )
        ]

    def index_objects(self) -> Any:
        records = self.build_index_records()
        if not records:
            return []

        if self.config.overwrite_db_collection:
            self.vector_db.build_db(overwrite=True)

        return self.vector_db.insert(
            records=records,
            build_if_missing=not self.config.overwrite_db_collection,
        )

    def vector_search(
        self,
        phrase: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        resolved_limit = self.config.search_limit if limit is None else limit
        query_vector = self.embeddings_model.embed_query(phrase)
        return self.vector_db.search(
            query_vector=query_vector,
            limit=resolved_limit,
        )

    def bm25_search(
        self,
        phrase: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        resolved_limit = self.config.search_limit if limit is None else limit
        return self.vector_db.bm25_search(
            query_text=phrase,
            limit=resolved_limit,
        )

    def search(
        self,
        phrase: str,
        limit: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        normalized_phrase = phrase.strip()
        if not normalized_phrase:
            raise ValueError("`phrase` must not be empty.")

        return {
            "vector_results": self.vector_search(normalized_phrase, limit=limit),
            "bm25_results": self.bm25_search(normalized_phrase, limit=limit),
        }

    def link_entities(
        self,
        phrase: str,
        class_hints: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        normalized_phrase = phrase.strip()
        if not normalized_phrase:
            return []

        resolved_limit = self.config.search_limit if limit is None else limit
        search_results = self.search(normalized_phrase, limit=resolved_limit)

        combined_results: List[Dict[str, Any]] = []
        seen_objects = set()

        for source_name in ("vector_results", "bm25_results"):
            for result in search_results.get(source_name, []):
                object_name = str(result.get("object_name", "")).strip()
                if not object_name or object_name in seen_objects:
                    continue

                if not self._result_matches_class_hints(result, class_hints):
                    continue

                linked_result = dict(result)
                linked_result["object_uri"] = self._resolve_object_uri(object_name)
                linked_result["source"] = source_name
                combined_results.append(linked_result)
                seen_objects.add(object_name)

        return combined_results

    def _resolve_object_uri(self, object_name: str) -> str:
        resolve_curie = getattr(self.graph_manager, "resolve_curie", None)
        if callable(resolve_curie):
            try:
                return str(resolve_curie(object_name, allow_bare=True))
            except Exception:
                return object_name

        return object_name

    def _result_matches_class_hints(
        self,
        result: Dict[str, Any],
        class_hints: Optional[List[str]] = None,
    ) -> bool:
        normalized_hints = [
            hint.strip().lower()
            for hint in class_hints or []
            if isinstance(hint, str) and hint.strip()
        ]
        if not normalized_hints:
            return True

        haystack_parts = [
            str(result.get("object_name", "")),
            str(result.get("object_description", "")),
            json.dumps(result.get("metadata", {}), sort_keys=True),
        ]
        haystack = " ".join(haystack_parts).lower()

        for hint in normalized_hints:
            formatted_hint = self._format_term(hint).lower()
            if hint in haystack or formatted_hint in haystack:
                return True

        return False

    def _format_term(self, value: str) -> str:
        if not value:
            return value

        formatted_value = value.strip().replace("\n", " ")
        reverse_curie = getattr(self.graph_manager, "reverse_curie", None)
        if callable(reverse_curie):
            try:
                return reverse_curie(formatted_value)
            except Exception:
                return formatted_value

        return formatted_value

    @staticmethod
    def _render_query(query_template: str, **kwargs: str) -> str:
        rendered_query = query_template
        for key, value in kwargs.items():
            rendered_query = rendered_query.replace(f"{{{key}}}", str(value))

        return rendered_query
 
