from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SKOS


DEFAULT_NAMESPACES: dict[str, str] = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms#",
    "prov": "http://www.w3.org/ns/prov#",
    "provone": "http://purl.dataone.org/provone/2015/01/15/ontology#",
    "sio": "http://semanticscience.org/resource/",
    "eo": "https://purl.org/heals/eo#",
    "workflow": "http://www.semanticweb.org/acer/ontologies/2026/1/WorkFlow/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

LABEL_PREDICATES = {
    str(RDFS.label),
    str(SKOS.prefLabel),
}
DESCRIPTION_PREDICATES = {
    "http://purl.org/dc/elements/1.1/description",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2004/02/skos/core#definition",
    "http://www.w3.org/ns/prov#definition",
}
IDENTIFIER_PREDICATES = {
    "http://purl.org/dc/terms#identifier",
}


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(content.encode("utf-8")).hexdigest()


def _clean_literal_text(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    cleaned = re.sub(r"@[A-Za-z-]+(?:\^\^<[^>]+>)?$", "", cleaned)
    cleaned = re.sub(r"\^\^<[^>]+>$", "", cleaned)
    return cleaned.strip()


def _humanize_identifier(value: str) -> str:
    text = value.replace("_", " ").replace("-", " ")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\bid\b\s+\d+(?:\s+\d+)*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _local_name(iri: str) -> str:
    if "#" in iri:
        return iri.rsplit("#", 1)[1]
    if "/" in iri:
        return iri.rstrip("/").rsplit("/", 1)[1]
    return iri


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _truncate(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


@dataclass(slots=True)
class EvidenceTriple:
    subject_id: str
    subject_label: str
    predicate_id: str
    predicate_label: str
    object_id: Optional[str]
    object_label: str
    object_is_literal: bool
    direction: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KGDocument:
    iri: str
    id: str
    label: str
    title: str
    text: str
    content: str
    types: list[str] = field(default_factory=list)
    evidence: list[EvidenceTriple] = field(default_factory=list)
    extracted_entities: list[str] = field(default_factory=list)
    extracted_triples: list[list[str]] = field(default_factory=list)

    def corpus_record(self, idx: int) -> dict[str, Any]:
        return {
            "title": self.title,
            "text": self.text,
            "idx": idx,
            "iri": self.iri,
            "id": self.id,
            "label": self.label,
            "types": self.types,
        }

    def openie_record(self) -> dict[str, Any]:
        return {
            "idx": compute_mdhash_id(self.content, "chunk-"),
            "passage": self.content,
            "extracted_entities": self.extracted_entities,
            "extracted_triples": self.extracted_triples,
        }


class KGObjectDocumentBuilder:
    def __init__(
        self,
        kg_path: str | Path,
        ontology_path: str | Path,
        metadata_path: str | Path | None = None,
        *,
        max_literal_chars: Optional[int] = 800,
    ) -> None:
        self.kg_path = Path(kg_path)
        self.ontology_path = Path(ontology_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.max_literal_chars = max_literal_chars

        self.kg = Graph()
        self.kg.parse(self.kg_path, format="turtle")

        self.ontology = Graph()
        self.ontology.parse(self.ontology_path, format="turtle")

        self.namespaces = self._load_namespaces()

    def build(self, max_objects: Optional[int] = None) -> list[KGDocument]:
        subjects = sorted(
            {
                str(subject)
                for subject in self.kg.subjects()
                if isinstance(subject, URIRef)
            }
        )
        if max_objects is not None and max_objects > 0:
            subjects = subjects[:max_objects]
        return [self._build_document(subject_iri) for subject_iri in subjects]

    def _build_document(self, subject_iri: str) -> KGDocument:
        subject = URIRef(subject_iri)
        subject_id = self.to_curie(subject_iri)
        subject_label = self._entity_label(subject_iri)
        subject_phrase = self._entity_phrase(subject_iri)

        types: list[str] = []
        property_lines: list[str] = []
        evidence: list[EvidenceTriple] = []
        entities = [subject_label, subject_id]
        triples: list[list[str]] = []

        for predicate, obj in sorted(
            self.kg.predicate_objects(subject),
            key=lambda item: (str(item[0]), str(item[1])),
        ):
            predicate_iri = str(predicate)
            predicate_id = self.to_curie(predicate_iri)
            predicate_label = self._predicate_label(predicate_iri)
            object_is_literal = isinstance(obj, Literal)
            object_iri = str(obj) if isinstance(obj, URIRef) else None
            object_id = self.to_curie(object_iri) if object_iri else None
            object_label = (
                self._entity_label(object_iri)
                if object_iri
                else _truncate(_clean_literal_text(str(obj)), self.max_literal_chars)
            )
            object_phrase = (
                self._entity_phrase(object_iri)
                if object_iri
                else object_label
            )

            entities.extend([object_label, object_id or ""])

            if predicate_iri == str(RDF.type) and object_id:
                types.append(object_id)
                triples.append([subject_phrase, "has type", object_phrase])
                continue

            property_lines.append(
                f"- {predicate_id} ({predicate_label}): "
                f"{object_id or json.dumps(object_label, ensure_ascii=True)}"
                + (f" [{object_label}]" if object_id else "")
            )
            triples.append([subject_phrase, predicate_label or predicate_id, object_phrase])
            evidence.append(
                EvidenceTriple(
                    subject_id=subject_id,
                    subject_label=subject_label,
                    predicate_id=predicate_id,
                    predicate_label=predicate_label,
                    object_id=object_id,
                    object_label=object_label,
                    object_is_literal=object_is_literal,
                    direction="outgoing",
                )
            )

        title = f"{subject_id} [{subject_label}]"
        text_parts = [f"Object: {subject_id} [{subject_label}]."]
        unique_types = _unique(types)
        if unique_types:
            text_parts.append("Types: " + ", ".join(unique_types) + ".")
        if property_lines:
            text_parts.append("Properties:\n" + "\n".join(property_lines))

        text = "\n".join(text_parts)
        content = f"{title}\n{text}"

        return KGDocument(
            iri=subject_iri,
            id=subject_id,
            label=subject_label,
            title=title,
            text=text,
            content=content,
            types=unique_types,
            evidence=evidence,
            extracted_entities=_unique(entities),
            extracted_triples=triples,
        )

    def _load_namespaces(self) -> dict[str, str]:
        namespaces = dict(DEFAULT_NAMESPACES)
        for prefix, uri in self.kg.namespaces():
            if prefix and uri:
                namespaces.setdefault(prefix, str(uri))
        for prefix, uri in self.ontology.namespaces():
            if prefix and uri:
                namespaces.setdefault(prefix, str(uri))
        if self.metadata_path and self.metadata_path.is_file():
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            for prefix, values in metadata.get("namespaces", {}).items():
                if not values:
                    continue
                uri = values[0] if isinstance(values, list) else values
                if isinstance(uri, str) and uri:
                    namespaces.setdefault(prefix, uri)
        return namespaces

    def _entity_label(self, iri: str) -> str:
        subject = URIRef(iri)
        for predicate_iri in LABEL_PREDICATES:
            for literal in self.kg.objects(subject, URIRef(predicate_iri)):
                if isinstance(literal, Literal):
                    cleaned = _clean_literal_text(str(literal))
                    if cleaned:
                        return _humanize_identifier(cleaned)
        for predicate_iri in IDENTIFIER_PREDICATES:
            for literal in self.kg.objects(subject, URIRef(predicate_iri)):
                if isinstance(literal, Literal):
                    cleaned = _clean_literal_text(str(literal))
                    if cleaned:
                        return _humanize_identifier(cleaned)
        return _humanize_identifier(_local_name(iri)) or self.to_curie(iri)

    def _entity_phrase(self, iri: str) -> str:
        entity_id = self.to_curie(iri)
        label = self._entity_label(iri)
        if label == entity_id:
            return label
        return f"{label} ({entity_id})"

    def _predicate_label(self, iri: str) -> str:
        subject = URIRef(iri)
        for predicate_iri in LABEL_PREDICATES | DESCRIPTION_PREDICATES:
            for literal in self.ontology.objects(subject, URIRef(predicate_iri)):
                if isinstance(literal, Literal):
                    cleaned = _clean_literal_text(str(literal))
                    if cleaned:
                        return _humanize_identifier(cleaned)
        return _humanize_identifier(_local_name(iri)) or self.to_curie(iri)

    def to_curie(self, iri: str) -> str:
        for prefix, namespace in sorted(
            self.namespaces.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        ):
            if iri.startswith(namespace):
                local_name = iri[len(namespace):]
                if re.match(r"^[A-Za-z][\w.-]*:", local_name):
                    return local_name
                return f"{prefix}:{local_name}"
        return iri


def build_kg_documents(
    kg_path: str | Path,
    ontology_path: str | Path,
    metadata_path: str | Path | None = None,
    *,
    max_literal_chars: Optional[int] = 800,
    max_objects: Optional[int] = None,
) -> list[KGDocument]:
    builder = KGObjectDocumentBuilder(
        kg_path=kg_path,
        ontology_path=ontology_path,
        metadata_path=metadata_path,
        max_literal_chars=max_literal_chars,
    )
    return builder.build(max_objects=max_objects)


def write_hipporag_corpus(documents: list[KGDocument], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [doc.corpus_record(idx) for idx, doc in enumerate(documents)]
    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return output_path


def write_hipporag_openie(documents: list[KGDocument], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    docs = [doc.openie_record() for doc in documents]
    entity_lengths = [
        len(entity)
        for doc in docs
        for entity in doc["extracted_entities"]
    ]
    entity_word_lengths = [
        len(entity.split())
        for doc in docs
        for entity in doc["extracted_entities"]
    ]
    payload = {
        "docs": docs,
        "avg_ent_chars": round(sum(entity_lengths) / len(entity_lengths), 4)
        if entity_lengths
        else 0,
        "avg_ent_words": round(sum(entity_word_lengths) / len(entity_word_lengths), 4)
        if entity_word_lengths
        else 0,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return output_path


def documents_by_content(documents: list[KGDocument]) -> dict[str, KGDocument]:
    return {document.content: document for document in documents}

