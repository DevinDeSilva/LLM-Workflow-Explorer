from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from baselines.HippoRAGKG import KGDocument, build_kg_documents


def _quote_upper(value: str) -> str:
    cleaned = " ".join(str(value).split()).strip().strip('"')
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", cleaned)
    return f'"{cleaned.upper()}"'


def _entity_display_name(entity_id: str, label: str) -> str:
    label = label.strip()
    if not label or label == entity_id:
        return entity_id
    return f"{label} ({entity_id})"


def _literal_entity_name(value: str) -> str:
    cleaned = " ".join(str(value).split()).strip()
    return cleaned[:160] if len(cleaned) > 160 else cleaned


@dataclass(slots=True)
class HyperGraphKG:
    chunks: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    hyperedges: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "entities": self.entities,
            "hyperedges": self.hyperedges,
        }


class HyperGraphKGBuilder:
    def __init__(self, documents: list[KGDocument]) -> None:
        self.documents = documents
        self.documents_by_id = {doc.id: doc for doc in documents}

    def build(self) -> HyperGraphKG:
        chunks = [
            {
                "source_id": doc.id,
                "content": doc.content,
            }
            for doc in self.documents
        ]

        entities: dict[str, dict[str, Any]] = {}
        hyperedges: list[dict[str, Any]] = []

        for doc in self.documents:
            subject_name = self._entity_name(doc.id, doc.label)
            self._add_entity(
                entities,
                entity_name=subject_name,
                entity_id=doc.id,
                label=doc.label,
                entity_type=self._primary_type(doc.types),
                description=doc.content,
                source_id=doc.id,
            )

            for evidence in doc.evidence:
                if evidence.object_id:
                    object_doc = self.documents_by_id.get(evidence.object_id)
                    object_name = self._entity_name(
                        evidence.object_id,
                        evidence.object_label,
                    )
                    object_type = self._primary_type(object_doc.types) if object_doc else "URI"
                    object_description = object_doc.content if object_doc else evidence.object_label
                    self._add_entity(
                        entities,
                        entity_name=object_name,
                        entity_id=evidence.object_id,
                        label=evidence.object_label,
                        entity_type=object_type,
                        description=object_description,
                        source_id=doc.id,
                    )
                else:
                    object_name = _quote_upper(_literal_entity_name(evidence.object_label))
                    self._add_entity(
                        entities,
                        entity_name=object_name,
                        entity_id=None,
                        label=evidence.object_label,
                        entity_type="LITERAL",
                        description=(
                            f"Literal value connected to {doc.label} via "
                            f"{evidence.predicate_label}: {evidence.object_label}"
                        ),
                        source_id=doc.id,
                    )

                hyperedge_text = (
                    f"{evidence.predicate_label}: {doc.label} "
                    f"({doc.id}) -> {evidence.object_label}"
                    + (f" ({evidence.object_id})" if evidence.object_id else "")
                )
                hyperedges.append(
                    {
                        "hyperedge_name": f"<hyperedge>{hyperedge_text}",
                        "description": hyperedge_text,
                        "predicate_id": evidence.predicate_id,
                        "predicate_label": evidence.predicate_label,
                        "source_id": doc.id,
                        "weight": 1.0,
                        "entity_names": [subject_name, object_name],
                    }
                )

        return HyperGraphKG(
            chunks=chunks,
            entities=list(entities.values()),
            hyperedges=hyperedges,
        )

    def _entity_name(self, entity_id: str, label: str) -> str:
        return _quote_upper(_entity_display_name(entity_id, label))

    def _primary_type(self, types: list[str]) -> str:
        return types[0] if types else "UNKNOWN"

    def _add_entity(
        self,
        entities: dict[str, dict[str, Any]],
        *,
        entity_name: str,
        entity_id: Optional[str],
        label: str,
        entity_type: str,
        description: str,
        source_id: str,
    ) -> None:
        existing = entities.get(entity_name)
        if existing is None:
            entities[entity_name] = {
                "entity_name": entity_name,
                "entity_id": entity_id,
                "label": label,
                "entity_type": entity_type,
                "description": description,
                "source_id": source_id,
            }
            return

        if source_id not in existing["source_id"].split("<SEP>"):
            existing["source_id"] += f"<SEP>{source_id}"
        if description and description not in existing["description"]:
            existing["description"] += f"<SEP>{description}"


def build_hypergraph_kg(
    kg_path: str | Path,
    ontology_path: str | Path,
    metadata_path: str | Path | None = None,
    *,
    max_literal_chars: Optional[int] = 800,
    max_objects: Optional[int] = None,
) -> HyperGraphKG:
    documents = build_kg_documents(
        kg_path=kg_path,
        ontology_path=ontology_path,
        metadata_path=metadata_path,
        max_literal_chars=max_literal_chars,
        max_objects=max_objects,
    )
    return HyperGraphKGBuilder(documents).build()


def write_hypergraph_kg(hypergraph_kg: HyperGraphKG, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(hypergraph_kg.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return output_path

