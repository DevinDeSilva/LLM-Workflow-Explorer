from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

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
        object_classes = self._extract_object_classes(properties)

        return self._build_object_description(
            object_name=object_name,
            properties=properties,
            object_classes=object_classes,
        )

    def get_object_classes(self, object_uri: str) -> List[str]:
        properties = self.get_object_properties(object_uri)

        return self._extract_object_classes(properties)

    def _build_object_description(
        self,
        object_name: str,
        properties: List[Dict[str, str]],
        object_classes: Optional[List[str]] = None,
    ) -> str:
        resolved_object_classes = object_classes or []

        description_parts = [f"Object: {object_name}."]
        if not properties:
            return " ".join(description_parts)

        property_lines: List[str] = []

        for property_row in sorted(
            properties,
            key=lambda row: (row["predicate"], row["value"]),
        ):
            predicate = property_row["predicate"]
            value = property_row["value"]
            formatted_value = self._format_term(value)

            if predicate == str(RDF.type):
                continue

            property_lines.append(f"{self._format_term(predicate)}: {formatted_value}")

        if resolved_object_classes:
            description_parts.append(
                "Types: " + ", ".join(resolved_object_classes) + "."
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

        object_records: List[Dict[str, Any]] = []
        descriptions: List[str] = []

        for object_uri in object_uris:
            object_name = self._format_term(object_uri)
            properties = self.get_object_properties(object_uri)
            object_class = self._extract_object_classes(properties)
            object_description = self._build_object_description(
                object_name=object_name,
                properties=properties,
                object_classes=object_class,
            )

            descriptions.append(object_description)
            object_records.append(
                {
                    "object_name": object_name,
                    "object_class": object_class,
                    "metadata": {},
                    "object_description": object_description,
                }
            )

        vectors = self.embeddings_model.embed_documents(descriptions)

        return [
            {
                "object_name": record["object_name"],
                "object_class": record["object_class"],
                "object_vector": object_vector,
                "metadata": record["metadata"],
                "object_description": record["object_description"],
            }
            for record, object_vector in zip(
                object_records,
                vectors,
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

    def link_entities_from_phrases(
        self,
        phrases: Sequence[str],
        class_hints: Optional[List[str]] = None,
        limit: Optional[int] = None,
        per_phrase_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        normalized_phrases = [
            str(phrase).strip()
            for phrase in phrases
            if str(phrase).strip()
        ]
        if not normalized_phrases:
            return []

        resolved_per_phrase_limit = (
            self.config.search_limit
            if per_phrase_limit is None
            else per_phrase_limit
        )
        combined_results: List[Dict[str, Any]] = []
        seen_object_uris = set()

        for phrase in normalized_phrases:
            linked_entities = self.link_entities(
                phrase,
                class_hints=class_hints,
                limit=resolved_per_phrase_limit,
            )
            for entity in linked_entities:
                object_key = str(
                    entity.get("object_uri")
                    or entity.get("object_name")
                    or ""
                ).strip()
                if not object_key or object_key in seen_object_uris:
                    continue

                combined_results.append(entity)
                seen_object_uris.add(object_key)
                if limit is not None and len(combined_results) >= limit:
                    return combined_results

        return combined_results

    def get_objects_of_class(
        self,
        class_uri: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        normalized_class_uri = str(class_uri).strip()
        if not normalized_class_uri:
            return []

        resolved_class_uri = self.graph_manager.resolve_curie(
            normalized_class_uri,
            allow_bare=True,
        )
        query = """
        SELECT DISTINCT ?object WHERE {
            ?object a <{class_uri}> .
            FILTER(isIRI(?object))
        }
        ORDER BY ?object
        """
        query_df = self.graph_manager.query(
            self._render_query(query, class_uri=resolved_class_uri),
            add_header_tail=False,
        )
        if query_df.empty or "object" not in query_df.columns:
            return []

        object_uris = list(dict.fromkeys(query_df["object"].tolist()))
        if limit is not None:
            object_uris = object_uris[:limit]

        return [
            {
                "object_uri": object_uri,
                "object_name": self._format_term(object_uri),
                "object_class": [self._format_term(normalized_class_uri)],
                "object_description": self.build_object_description(object_uri),
                "source": "class-query",
            }
            for object_uri in object_uris
        ]

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

        raw_result_classes = result.get("object_class", [])
        normalized_result_classes = set(
            self._normalize_class_terms(raw_result_classes)
        )
        if normalized_result_classes:
            for hint in normalized_hints:
                formatted_hint = self._format_term(hint).lower()
                if (
                    hint in normalized_result_classes
                    or formatted_hint in normalized_result_classes
                ):
                    return True

        haystack_parts = [
            str(result.get("object_name", "")),
            str(result.get("object_description", "")),
            json.dumps(result.get("metadata", {}), sort_keys=True),
            json.dumps(raw_result_classes, sort_keys=True),
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

    def _extract_object_classes(
        self,
        properties: List[Dict[str, str]],
    ) -> List[str]:
        object_classes = [
            self._format_term(property_row["value"])
            for property_row in properties
            if property_row.get("predicate") == str(RDF.type)
        ]

        return list(dict.fromkeys(sorted(object_classes)))

    def _normalize_class_terms(
        self,
        values: Optional[Sequence[str] | str],
    ) -> List[str]:
        if values is None:
            return []

        if isinstance(values, str):
            candidate_values = [values]
        else:
            candidate_values = [str(value) for value in values]

        normalized_terms: List[str] = []
        seen_terms = set()
        for value in candidate_values:
            stripped_value = value.strip()
            if not stripped_value:
                continue

            for candidate in (stripped_value, self._format_term(stripped_value)):
                normalized_candidate = candidate.lower()
                if normalized_candidate in seen_terms:
                    continue

                seen_terms.add(normalized_candidate)
                normalized_terms.append(normalized_candidate)

        return normalized_terms

    @staticmethod
    def _render_query(query_template: str, **kwargs: str) -> str:
        rendered_query = query_template
        for key, value in kwargs.items():
            rendered_query = rendered_query.replace(f"{{{key}}}", str(value))

        return rendered_query
 
