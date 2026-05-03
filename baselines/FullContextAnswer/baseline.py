from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SKOS
import dycomutils as common_utils
from src.utils.ontology_info_retriever2 import OntologyInfoRetriever


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
MODEL_PREDICATES = {
    "http://www.semanticweb.org/acer/ontologies/2026/1/WorkFlow/llm_model",
}


def _empty_token_usage() -> dict[str, Any]:
    return {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "models": {},
        "calls": [],
        "estimated": False,
        "source": "none",
    }


def _token_count_from_usage(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _stringify_for_token_count(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        prioritized_parts = [
            _stringify_for_token_count(value[key])
            for key in ("role", "content", "text", "reasoning_content")
            if key in value
        ]
        if prioritized_parts:
            return "\n".join(part for part in prioritized_parts if part)
        return json.dumps(value, default=str, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify_for_token_count(item) for item in value)
    return str(value)


def _count_text_tokens(value: Any, model: str = "") -> int:
    text = _stringify_for_token_count(value)
    if not text:
        return 0

    try:
        import tiktoken

        model_name = str(model).split("/", 1)[-1]
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(re.findall(r"\w+|[^\w\s]", text))


def _plain_usage_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        try:
            return dict(value.model_dump())
        except Exception:
            return {}
    try:
        return dict(value)
    except Exception:
        return {}


def _normalize_usage(usage: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = _token_count_from_usage(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _token_count_from_usage(
        usage,
        "completion_tokens",
        "output_tokens",
    )
    total_tokens = _token_count_from_usage(usage, "total_tokens")
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens

    normalized: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    for details_key in (
        "prompt_tokens_details",
        "completion_tokens_details",
        "input_token_details",
        "output_token_details",
    ):
        if details_key in usage:
            normalized[details_key] = usage[details_key]
    return normalized


def _has_token_counts(usage: dict[str, Any]) -> bool:
    return bool(
        _token_count_from_usage(
            usage,
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "input_tokens",
            "output_tokens",
        )
    )


def _unique(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _clean_literal_text(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    cleaned = re.sub(r"@[A-Za-z-]+(?:\^\^<[^>]+>)?$", "", cleaned)
    cleaned = re.sub(r"\^\^<[^>]+>$", "", cleaned)
    return cleaned.strip()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


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


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@dataclass(slots=True)
class RelevantEntity:
    id: str
    label: str
    types: list[str]
    score: float


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


@dataclass(slots=True)
class ExplanationResponse:
    answer: str
    relevant_entities: list[RelevantEntity]
    evidence: list[EvidenceTriple] = field(default_factory=list)
    token_usage: dict[str, Any] = field(default_factory=_empty_token_usage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "relevant_entities": [asdict(entity) for entity in self.relevant_entities],
            "evidence": [asdict(triple) for triple in self.evidence],
            "token_usage": self.token_usage,
        }


@dataclass(slots=True)
class EntityRecord:
    iri: str
    curie: str
    label: str
    types: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    normalized_label: str = ""


@dataclass(slots=True)
class TripleRecord:
    subject_iri: str
    subject_id: str
    subject_label: str
    predicate_iri: str
    predicate_id: str
    predicate_label: str
    object_iri: Optional[str]
    object_id: Optional[str]
    object_label: str
    object_is_literal: bool
    context_line: str


class EvidenceSelection(BaseModel):
    subject: str = Field(default="")
    predicate: str = Field(default="")
    object: str = Field(default="")


class LLMAnswerPayload(BaseModel):
    answer: str = Field(default="")
    relevant_entities: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSelection] = Field(default_factory=list)


class FullContextAnswerBaseline:
    """
    LLM-backed workflow explanation baseline that sends the application
    description, schema summary, and the full execution KG at inference time.
    """

    def __init__(
        self,
        kg_path: str | Path,
        ontology_path: str | Path,
        schema_json_path: str | Path,
        metadata_path: str | Path | None = None,
        *,
        application_description: str = "",
        llm: Any | None = None,
        llm_type: str = "openai",
        llm_library: str = "langchain",
        llm_config: Optional[dict[str, Any]] = None,
        max_relevant_entities: int = 6,
        max_evidence: int = 8,
        max_triples_in_prompt: int | None = None,
    ) -> None:
        self.kg_path = Path(kg_path)
        self.ontology_path = Path(ontology_path)
        self.schema_json_path = Path(schema_json_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.application_description = application_description.strip()
        self.max_relevant_entities = max_relevant_entities
        self.max_evidence = max_evidence
        self.max_triples_in_prompt = max_triples_in_prompt

        self.kg = Graph()
        self.kg.parse(self.kg_path, format="turtle")

        self.ontology = Graph()
        self.ontology.parse(self.ontology_path, format="turtle")

        self.namespaces = self._load_namespaces()
        self.schema = self._load_schema()
        self.entities = self._build_entity_index()
        self.predicates = self._build_predicate_index()
        self.triples = self._build_triples()
        self.schema_context = self._build_schema_context()
        self.graph_context = self._build_graph_context()
        self.application_context = self._derive_application_context()
        self.onto_retriever = OntologyInfoRetriever(
            "schema/extracted_ontology_triples.csv",
            "schema/schemaV2.json"
        )
        self.schema_context = self.onto_retriever.format_schema_prompt()
        self.graph_context = self.dump_graph()
        self.last_token_usage = _empty_token_usage()

        if llm is None:
            from src.llm import LLM

            self.llm = LLM(llm_type, llm_library, **(llm_config or {}))
        else:
            self.llm = llm
            
    def dump_graph(self):
        rep_str = common_utils.serialization.load_text(str(self.kg_path))
        return rep_str

    def request(self, user_query: str) -> dict[str, Any]:
        response = self.answer(user_query)
        return response.to_dict()

    def answer(self, user_query: str) -> ExplanationResponse:
        question = user_query.strip()
        if not question:
            return ExplanationResponse(
                answer="The question is empty, so there is no answer to return.",
                relevant_entities=[],
                evidence=[],
                token_usage=_empty_token_usage(),
            )

        payload = self._query_llm(question)
        answer = payload.answer.strip() or "I could not produce an answer from the provided workflow context."
        relevant_entities = self._resolve_relevant_entities(payload.relevant_entities, answer)
        evidence = self._resolve_evidence(payload.evidence)
        relevant_entities = relevant_entities[: self.max_relevant_entities]
        answer = self._attach_entity_citations(answer, relevant_entities)

        return ExplanationResponse(
            answer=answer,
            relevant_entities=relevant_entities,
            evidence=evidence[: self.max_evidence],
            token_usage=self.last_token_usage,
        )

    def _attach_entity_citations(
        self,
        answer: str,
        relevant_entities: list[RelevantEntity],
    ) -> str:
        cleaned_answer = re.sub(r"\s*<cite,\s*id=\d+>.*?</cite>", "", answer).strip()
        if not relevant_entities:
            return cleaned_answer

        citations = " ".join(
            f"<cite, id={index}>{entity.label}</cite>"
            for index, entity in enumerate(relevant_entities)
        )
        if not cleaned_answer:
            return citations
        return f"{cleaned_answer} {citations}"

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

    def _load_schema(self) -> dict[str, Any]:
        return json.loads(self.schema_json_path.read_text(encoding="utf-8"))

    def _derive_application_context(self) -> str:
        if self.application_description:
            return self.application_description

        if self.metadata_path and self.metadata_path.is_file():
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            generated_by = metadata.get("generatedBy", [])
            if isinstance(generated_by, list) and generated_by:
                return f"Workflow provenance generated by {generated_by[0]}."

        return "Workflow provenance graph for an application execution."

    def _build_entity_index(self) -> dict[str, EntityRecord]:
        entities: dict[str, EntityRecord] = {}
        for subject, predicate, obj in self.kg:
            if isinstance(subject, URIRef):
                subject_record = entities.setdefault(
                    str(subject),
                    EntityRecord(
                        iri=str(subject),
                        curie=self.to_curie(str(subject)),
                        label="",
                    ),
                )
                predicate_iri = str(predicate)
                if predicate_iri == str(RDF.type) and isinstance(obj, URIRef):
                    subject_record.types.append(self.to_curie(str(obj)))
                elif isinstance(obj, Literal):
                    cleaned_literal = _clean_literal_text(str(obj))
                    if not cleaned_literal:
                        continue
                    if predicate_iri in LABEL_PREDICATES:
                        subject_record.aliases.append(cleaned_literal)
                    elif predicate_iri in DESCRIPTION_PREDICATES:
                        subject_record.descriptions.append(cleaned_literal)
                    elif predicate_iri in IDENTIFIER_PREDICATES:
                        subject_record.identifiers.append(cleaned_literal)

            if isinstance(obj, URIRef):
                entities.setdefault(
                    str(obj),
                    EntityRecord(
                        iri=str(obj),
                        curie=self.to_curie(str(obj)),
                        label="",
                    ),
                )

        for record in entities.values():
            record.types = _unique(record.types)
            record.aliases = _unique(record.aliases)
            record.descriptions = _unique(record.descriptions)
            record.identifiers = _unique(record.identifiers)
            record.label = self._derive_entity_label(record)
            record.normalized_label = _normalize_text(record.label)
        return entities

    def _build_predicate_index(self) -> dict[str, dict[str, str]]:
        predicates: dict[str, dict[str, str]] = {}
        predicate_terms = set(self.schema.get("object_properties", {}).keys())
        predicate_terms.update(self.to_curie(str(predicate)) for _, predicate, _ in self.kg)

        for predicate_id in predicate_terms:
            predicate_iri = self.expand_curie(predicate_id)
            label = self._first_literal(self.ontology, predicate_iri, LABEL_PREDICATES)
            description = self._first_literal(self.ontology, predicate_iri, DESCRIPTION_PREDICATES)
            predicates[predicate_iri] = {
                "id": self.to_curie(predicate_iri),
                "label": label or _humanize_identifier(_local_name(predicate_iri)),
                "description": description or "",
            }
        return predicates

    def _build_triples(self) -> list[TripleRecord]:
        records: list[TripleRecord] = []
        for subject, predicate, obj in self.kg:
            if not isinstance(subject, URIRef):
                continue

            subject_iri = str(subject)
            predicate_iri = str(predicate)
            subject_id = self.to_curie(subject_iri)
            predicate_id = self.to_curie(predicate_iri)
            subject_label = self.entities.get(subject_iri, EntityRecord(subject_iri, subject_id, subject_id)).label
            predicate_label = self.predicates.get(
                predicate_iri,
                {"label": _humanize_identifier(_local_name(predicate_iri)) or predicate_id},
            )["label"]

            object_iri: Optional[str] = None
            object_id: Optional[str] = None
            object_is_literal = isinstance(obj, Literal)
            if isinstance(obj, URIRef):
                object_iri = str(obj)
                object_id = self.to_curie(object_iri)
                object_label = self.entities.get(object_iri, EntityRecord(object_iri, object_id, object_id)).label
            else:
                object_label = _clean_literal_text(str(obj))

            context_line = (
                f"{subject_id} [{subject_label}] "
                f"{predicate_id} [{predicate_label}] "
                f"{object_id if object_id else json.dumps(object_label, ensure_ascii=False)}"
            )
            if object_id:
                context_line += f" [{object_label}]"

            records.append(
                TripleRecord(
                    subject_iri=subject_iri,
                    subject_id=subject_id,
                    subject_label=subject_label,
                    predicate_iri=predicate_iri,
                    predicate_id=predicate_id,
                    predicate_label=predicate_label,
                    object_iri=object_iri,
                    object_id=object_id,
                    object_label=object_label,
                    object_is_literal=object_is_literal,
                    context_line=context_line,
                )
            )

        records.sort(key=lambda record: record.context_line)
        return records

    def _build_schema_context_low(self) -> str:
        lines = ["Classes:"]
        for class_id in self.schema.get("classes", []):
            iri = self.expand_curie(class_id)
            label = self._first_literal(self.ontology, iri, LABEL_PREDICATES) or _humanize_identifier(_local_name(iri))
            description = self._first_literal(self.ontology, iri, DESCRIPTION_PREDICATES)
            line = f"- {class_id}: {label}"
            if description:
                line += f" | {description}"
            lines.append(line)

        lines.append("")
        lines.append("Object properties:")
        for property_id, details in sorted(self.schema.get("object_properties", {}).items()):
            predicate_iri = self.expand_curie(property_id)
            label = self.predicates.get(predicate_iri, {}).get("label") or _humanize_identifier(_local_name(predicate_iri))
            connections = details.get("connections", [])
            connection_text = ", ".join(
                f"{item.get('domain', '?')} -> {item.get('range', '?')}"
                for item in connections[:6]
            )
            lines.append(f"- {property_id}: {label}" + (f" | {connection_text}" if connection_text else ""))
        return "\n".join(lines)

    def _build_schema_context(self) -> str:
        lines = ["Classes:"]
        for class_id in self.schema.get("classes", []):
            iri = self.expand_curie(class_id)
            label = self._first_literal(self.ontology, iri, LABEL_PREDICATES) or _humanize_identifier(_local_name(iri))
            description = self._first_literal(self.ontology, iri, DESCRIPTION_PREDICATES)
            line = f"- {class_id}: {label}"
            if description:
                line += f" | {description}"
            lines.append(line)

        lines.append("")
        lines.append("Object properties:")
        for property_id, details in sorted(self.schema.get("object_properties", {}).items()):
            predicate_iri = self.expand_curie(property_id)
            label = self.predicates.get(predicate_iri, {}).get("label") or _humanize_identifier(_local_name(predicate_iri))
            connections = details.get("connections", [])
            connection_text = ", ".join(
                f"{item.get('domain', '?')} -> {item.get('range', '?')}"
                for item in connections[:6]
            )
            lines.append(f"- {property_id}: {label}" + (f" | {connection_text}" if connection_text else ""))
        return "\n".join(lines)

    def _build_graph_context(self) -> str:
        triples = self.triples
        if self.max_triples_in_prompt is not None:
            triples = triples[: self.max_triples_in_prompt]
        lines = [f"Total triples included: {len(triples)}"]
        lines.extend(f"- {triple.context_line}" for triple in triples)
        return "\n".join(lines)

    def _build_prompt(self, question: str) -> tuple[str, str]:
        namespaces = "\n".join(
            f"- {prefix}: {uri}"
            for prefix, uri in sorted(self.namespaces.items())
        )
        system_prompt = (
            "You answer workflow-explanation questions using only the provided application description, "
            "schema summary, and full provenance knowledge graph. "
            "Do not invent facts. If the graph is insufficient, say that directly."
        )
        prompt = (
            "Application description:\n"
            f"{self.application_context}\n\n"
            "Namespace prefixes:\n"
            f"{namespaces}\n\n"
            "Schema summary:\n"
            f"{self.schema_context}\n\n"
            "Full execution KG:\n"
            f"{self.graph_context}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Return a JSON object with these fields:\n"
            "- answer: concise natural-language answer grounded in the graph\n"
            "- relevant_entities: list of exact KG entity CURIEs or IRIs most central to the answer\n"
            "- evidence: list of supporting triples with fields subject, predicate, object using exact CURIEs, IRIs, or literal text from the graph\n"
            "Keep the answer concise and faithful to the provided graph."
        )
        return system_prompt, prompt

    def _langchain_messages(self, prompt: str, system_prompt: str) -> list[Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages: list[Any] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages

    def _response_text(self, response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        return _stringify_for_token_count(content)

    def _payload_text(self, payload: Any) -> str:
        if isinstance(payload, BaseModel):
            return payload.model_dump_json()
        if isinstance(payload, dict):
            return json.dumps(payload, default=str, ensure_ascii=False)
        return _stringify_for_token_count(payload)

    def _extract_usage_from_response(self, response: Any) -> dict[str, Any]:
        if response is None:
            return {}

        candidates: list[Any] = []
        response_metadata: Any = None
        additional_kwargs: Any = None
        if isinstance(response, dict):
            candidates.extend(
                [
                    response.get("usage_metadata"),
                    response.get("usage"),
                ]
            )
            response_metadata = response.get("response_metadata")
            additional_kwargs = response.get("additional_kwargs")
        else:
            candidates.extend(
                [
                    getattr(response, "usage_metadata", None),
                    getattr(response, "usage", None),
                ]
            )
            response_metadata = getattr(response, "response_metadata", None)
            additional_kwargs = getattr(response, "additional_kwargs", None)

        metadata = _plain_usage_dict(response_metadata)
        if metadata:
            candidates.extend(
                [
                    metadata.get("token_usage"),
                    metadata.get("usage"),
                    metadata,
                ]
            )

        kwargs = _plain_usage_dict(additional_kwargs)
        if kwargs:
            candidates.extend([kwargs.get("usage"), kwargs.get("token_usage")])

        for candidate in candidates:
            usage = _plain_usage_dict(candidate)
            if not usage:
                continue
            normalized = _normalize_usage(usage)
            if _has_token_counts(normalized):
                return normalized
        return {}

    def _extract_model_name(self, response: Any = None) -> str:
        if isinstance(response, dict):
            metadata = _plain_usage_dict(response.get("response_metadata"))
        else:
            metadata = _plain_usage_dict(getattr(response, "response_metadata", None))
        for key in ("model_name", "model", "model_id"):
            value = metadata.get(key)
            if value:
                return str(value)

        config = getattr(self.llm, "config", None)
        value = getattr(config, "model", None)
        if value:
            return str(value)

        client = getattr(self.llm, "llm", None)
        for attr in ("model_name", "model", "model_id"):
            value = getattr(client, attr, None)
            if value:
                return str(value)
        return "unknown"

    def _token_usage_for_call(
        self,
        *,
        system_prompt: str,
        prompt: str,
        completion_text: Any,
        response: Any = None,
    ) -> dict[str, Any]:
        model = self._extract_model_name(response)
        usage = self._extract_usage_from_response(response)
        estimated = False
        source = "provider"

        if not _has_token_counts(usage):
            prompt_tokens = _count_text_tokens([system_prompt, prompt], model=model)
            completion_tokens = _count_text_tokens(completion_text, model=model)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            estimated = True
            source = "estimate"

        return {
            "total_tokens": _token_count_from_usage(usage, "total_tokens"),
            "prompt_tokens": _token_count_from_usage(usage, "prompt_tokens"),
            "completion_tokens": _token_count_from_usage(usage, "completion_tokens"),
            "models": {model: usage},
            "calls": [
                {
                    "model": model,
                    "usage": usage,
                    "estimated": estimated,
                    "source": source,
                }
            ],
            "estimated": estimated,
            "source": source,
        }

    def _query_langchain_structured(
        self,
        *,
        prompt: str,
        system_prompt: str,
    ) -> LLMAnswerPayload | None:
        chat_model = getattr(self.llm, "llm", None)
        if chat_model is None or not hasattr(chat_model, "with_structured_output"):
            return None

        try:
            messages = self._langchain_messages(prompt, system_prompt)
            structured_llm = chat_model.with_structured_output(
                LLMAnswerPayload,
                method="function_calling",
                include_raw=True,
            )
            response = _run_async(structured_llm.ainvoke(messages))
            if not isinstance(response, dict):
                return None

            raw_response = response.get("raw")
            parsed = response.get("parsed")
            if parsed is None:
                parsing_error = response.get("parsing_error")
                if parsing_error:
                    raise parsing_error
                return None

            payload = LLMAnswerPayload.model_validate(parsed)
            self.last_token_usage = self._token_usage_for_call(
                system_prompt=system_prompt,
                prompt=prompt,
                completion_text=self._payload_text(payload),
                response=raw_response,
            )
            return payload
        except Exception:
            return None

    def _query_langchain_raw(
        self,
        *,
        prompt: str,
        system_prompt: str,
    ) -> LLMAnswerPayload | None:
        chat_model = getattr(self.llm, "llm", None)
        if chat_model is None or not hasattr(chat_model, "ainvoke"):
            return None

        try:
            response = _run_async(
                chat_model.ainvoke(self._langchain_messages(prompt, system_prompt))
            )
            raw_response = self._response_text(response)
            payload = self._parse_raw_response(raw_response)
            self.last_token_usage = self._token_usage_for_call(
                system_prompt=system_prompt,
                prompt=prompt,
                completion_text=raw_response,
                response=response,
            )
            return payload
        except Exception:
            return None

    def _query_llm(self, question: str) -> LLMAnswerPayload:
        self.last_token_usage = _empty_token_usage()
        system_prompt, prompt = self._build_prompt(question)

        direct_payload = self._query_langchain_structured(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        if direct_payload is not None:
            return direct_payload

        if hasattr(self.llm, "structured_generate"):
            try:
                response = _run_async(
                    self.llm.structured_generate(
                        prompt=prompt,
                        structure=LLMAnswerPayload,
                        system_prompt=system_prompt,
                    )
                )
                payload = LLMAnswerPayload.model_validate(response)
                self.last_token_usage = self._token_usage_for_call(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    completion_text=self._payload_text(payload),
                    response=response,
                )
                return payload
            except Exception:
                pass

        direct_payload = self._query_langchain_raw(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        if direct_payload is not None:
            return direct_payload

        raw_response = _run_async(self.llm.generate(prompt=prompt, system_prompt=system_prompt))
        payload = self._parse_raw_response(raw_response)
        self.last_token_usage = self._token_usage_for_call(
            system_prompt=system_prompt,
            prompt=prompt,
            completion_text=raw_response,
        )
        return payload

    def _parse_raw_response(self, raw_response: str) -> LLMAnswerPayload:
        text = raw_response.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1).strip()
        try:
            data = json.loads(text)
            return LLMAnswerPayload.model_validate(data)
        except Exception:
            return LLMAnswerPayload(answer=raw_response.strip())

    def _resolve_relevant_entities(self, references: list[str], answer_text: str) -> list[RelevantEntity]:
        resolved: list[RelevantEntity] = []
        seen: set[str] = set()

        for index, reference in enumerate(references):
            record = self._resolve_entity_reference(reference)
            if not record or record.curie in seen:
                continue
            seen.add(record.curie)
            resolved.append(
                RelevantEntity(
                    id=record.curie,
                    label=record.label,
                    types=record.types,
                    score=max(0.0, 1.0 - (index * 0.1)),
                )
            )

        if resolved:
            return resolved

        normalized_answer = _normalize_text(answer_text)
        fallback_records = sorted(
            (
                record
                for record in self.entities.values()
                if record.normalized_label and record.normalized_label in normalized_answer
            ),
            key=lambda record: len(record.normalized_label),
            reverse=True,
        )
        for index, record in enumerate(fallback_records[: self.max_relevant_entities]):
            if record.curie in seen:
                continue
            seen.add(record.curie)
            resolved.append(
                RelevantEntity(
                    id=record.curie,
                    label=record.label,
                    types=record.types,
                    score=max(0.0, 0.5 - (index * 0.05)),
                )
            )
        return resolved

    def _resolve_evidence(self, selections: list[EvidenceSelection]) -> list[EvidenceTriple]:
        evidence: list[EvidenceTriple] = []
        seen: set[tuple[str, str, Optional[str], str]] = set()

        for index, selection in enumerate(selections):
            triple = self._resolve_evidence_selection(selection)
            if triple is None:
                continue
            key = (triple.subject_id, triple.predicate_id, triple.object_id, triple.object_label)
            if key in seen:
                continue
            seen.add(key)
            triple.score = max(0.0, 1.0 - (index * 0.1))
            evidence.append(triple)

        return evidence

    def _resolve_evidence_selection(self, selection: EvidenceSelection) -> EvidenceTriple | None:
        subject = self._resolve_entity_reference(selection.subject)
        predicate_id, predicate_iri, predicate_label = self._resolve_predicate_reference(selection.predicate)

        if subject:
            object_entity = self._resolve_entity_reference(selection.object)
            literal_value = _clean_literal_text(selection.object)
            candidates = [
                triple
                for triple in self.triples
                if triple.subject_iri == subject.iri and triple.predicate_iri == predicate_iri
            ]
            if object_entity:
                for triple in candidates:
                    if triple.object_iri == object_entity.iri:
                        return self._to_evidence(triple)
            if literal_value:
                for triple in candidates:
                    if triple.object_is_literal and _normalize_text(triple.object_label) == _normalize_text(literal_value):
                        return self._to_evidence(triple)

        normalized_subject = _normalize_text(selection.subject)
        normalized_predicate = _normalize_text(selection.predicate)
        normalized_object = _normalize_text(selection.object)
        for triple in self.triples:
            if normalized_subject and normalized_subject not in {
                _normalize_text(triple.subject_id),
                _normalize_text(triple.subject_label),
            }:
                continue
            if normalized_predicate and normalized_predicate not in {
                _normalize_text(triple.predicate_id),
                _normalize_text(triple.predicate_label),
            }:
                continue
            object_options = {_normalize_text(triple.object_label)}
            if triple.object_id:
                object_options.add(_normalize_text(triple.object_id))
            if normalized_object and normalized_object not in object_options:
                continue
            return self._to_evidence(triple)

        if not subject:
            return None

        object_entity = self._resolve_entity_reference(selection.object)
        return EvidenceTriple(
            subject_id=subject.curie,
            subject_label=subject.label,
            predicate_id=predicate_id,
            predicate_label=predicate_label,
            object_id=object_entity.curie if object_entity else None,
            object_label=object_entity.label if object_entity else _clean_literal_text(selection.object),
            object_is_literal=object_entity is None,
            direction="outgoing",
            score=0.0,
        )

    def _to_evidence(self, triple: TripleRecord) -> EvidenceTriple:
        return EvidenceTriple(
            subject_id=triple.subject_id,
            subject_label=triple.subject_label,
            predicate_id=triple.predicate_id,
            predicate_label=triple.predicate_label,
            object_id=triple.object_id,
            object_label=triple.object_label,
            object_is_literal=triple.object_is_literal,
            direction="outgoing",
            score=0.0,
        )

    def _resolve_entity_reference(self, reference: str) -> EntityRecord | None:
        ref = reference.strip()
        if not ref:
            return None

        if ref in self.entities:
            return self.entities[ref]

        expanded = self.expand_curie(ref)
        if expanded in self.entities:
            return self.entities[expanded]

        normalized = _normalize_text(ref)
        for record in self.entities.values():
            if normalized in {
                _normalize_text(record.iri),
                _normalize_text(record.curie),
                record.normalized_label,
            }:
                return record
            if normalized and any(_normalize_text(alias) == normalized for alias in record.aliases):
                return record
        return None

    def _resolve_predicate_reference(self, reference: str) -> tuple[str, str, str]:
        ref = reference.strip()
        if not ref:
            return ("rdf:type", str(RDF.type), "type")

        if ref in self.predicates:
            info = self.predicates[ref]
            return (info["id"], ref, info["label"])

        expanded = self.expand_curie(ref)
        if expanded in self.predicates:
            info = self.predicates[expanded]
            return (info["id"], expanded, info["label"])

        normalized = _normalize_text(ref)
        for predicate_iri, info in self.predicates.items():
            if normalized in {_normalize_text(predicate_iri), _normalize_text(info["id"]), _normalize_text(info["label"])}:
                return (info["id"], predicate_iri, info["label"])

        return (self.to_curie(expanded), expanded, _humanize_identifier(_local_name(expanded)) or ref)

    def to_curie(self, iri: str) -> str:
        for prefix, namespace in sorted(self.namespaces.items(), key=lambda item: len(item[1]), reverse=True):
            if iri.startswith(namespace):
                return f"{prefix}:{iri[len(namespace):]}"
        return iri

    def expand_curie(self, value: str) -> str:
        if "://" in value:
            return value
        if ":" not in value:
            return value
        prefix, local_name = value.split(":", 1)
        namespace = self.namespaces.get(prefix)
        return f"{namespace}{local_name}" if namespace else value

    def _derive_entity_label(self, record: EntityRecord) -> str:
        if record.aliases:
            alias = record.aliases[0]
            if " " not in alias and any(char in alias for char in {"_", "-"}):
                humanized_alias = _humanize_identifier(alias)
                if humanized_alias:
                    return humanized_alias
            return alias
        if record.identifiers and any(entity_type == "provone:Execution" for entity_type in record.types):
            return f"execution {record.identifiers[0]}"
        model_name = self._literal_for_predicate(record.iri, MODEL_PREDICATES)
        if model_name:
            return f"{model_name} model"

        local_name = _local_name(record.iri)
        cleaned_local_name = re.sub(r"^[A-Za-z_]+-id_[0-9_]+-", "", local_name)
        if ":" in cleaned_local_name:
            cleaned_local_name = cleaned_local_name.split(":", 1)[1]
        humanized = _humanize_identifier(cleaned_local_name)
        return humanized or record.curie

    def _literal_for_predicate(self, subject_iri: str, predicate_iris: set[str]) -> str:
        subject = URIRef(subject_iri)
        for predicate_iri in predicate_iris:
            for literal in self.kg.objects(subject, URIRef(predicate_iri)):
                if isinstance(literal, Literal):
                    cleaned = _clean_literal_text(str(literal))
                    if cleaned:
                        return cleaned
        return ""

    def _first_literal(self, graph: Graph, subject_iri: str, predicate_iris: set[str]) -> str:
        subject = URIRef(subject_iri)
        for predicate_iri in predicate_iris:
            for literal in graph.objects(subject, URIRef(predicate_iri)):
                if isinstance(literal, Literal):
                    cleaned = _clean_literal_text(str(literal))
                    if cleaned:
                        return cleaned
        return ""
