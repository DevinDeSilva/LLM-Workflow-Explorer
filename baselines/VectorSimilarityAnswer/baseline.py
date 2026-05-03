from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field
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


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


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
class RetrievedObject:
    id: str
    label: str
    types: list[str]
    description: str
    score: float


@dataclass(slots=True)
class ObjectSegment:
    iri: str
    id: str
    label: str
    types: list[str]
    description: str
    evidence: list[EvidenceTriple]
    score: float = 0.0
    vector: list[float] = field(default_factory=list)


@dataclass(slots=True)
class ExplanationResponse:
    answer: str
    relevant_entities: list[RelevantEntity]
    evidence: list[EvidenceTriple] = field(default_factory=list)
    retrieved_objects: list[RetrievedObject] = field(default_factory=list)
    token_usage: dict[str, Any] = field(default_factory=_empty_token_usage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "relevant_entities": [asdict(entity) for entity in self.relevant_entities],
            "evidence": [asdict(triple) for triple in self.evidence],
            "retrieved_objects": [asdict(obj) for obj in self.retrieved_objects],
            "token_usage": self.token_usage,
        }


class LLMAnswerPayload(BaseModel):
    answer: str = Field(default="")


class VectorSimilarityAnswerBaseline:
    """
    Retrieval baseline that embeds per-object TTL segments and answers from the
    top vector-similar object descriptions.
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
        embeddings_model: Any | None = None,
        embedding_type: str = "openai",
        embedding_config: Optional[dict[str, Any]] = None,
        top_k: int = 5,
        max_evidence: int = 12,
    ) -> None:
        self.kg_path = Path(kg_path)
        self.ontology_path = Path(ontology_path)
        self.schema_json_path = Path(schema_json_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.application_description = application_description.strip()
        self.top_k = top_k
        self.max_evidence = max_evidence
        self.last_token_usage = _empty_token_usage()

        self.kg = Graph()
        self.kg.parse(self.kg_path, format="turtle")

        self.ontology = Graph()
        self.ontology.parse(self.ontology_path, format="turtle")

        self.namespaces = self._load_namespaces()

        self.llm = llm
        if self.llm is None:
            from src.llm import LLM

            self.llm = LLM(llm_type, llm_library, **(llm_config or {}))

        self.embeddings_model = embeddings_model
        if self.embeddings_model is None:
            from src.embeddings import Embeddings

            self.embeddings_model = Embeddings(
                embedding_type,
                **(embedding_config or {}),
            )

        self.segments = self._build_object_segments()
        self._embed_segments()

    def request(self, user_query: str) -> dict[str, Any]:
        return self.answer(user_query).to_dict()

    def answer(self, user_query: str) -> ExplanationResponse:
        question = user_query.strip()
        if not question:
            return ExplanationResponse(
                answer="The question is empty, so there is no answer to return.",
                relevant_entities=[],
                evidence=[],
                retrieved_objects=[],
                token_usage=_empty_token_usage(),
            )

        retrieved_segments = self.search(question, limit=self.top_k)
        answer = self._answer_with_llm(question, retrieved_segments)
        if not answer:
            answer = self._fallback_answer(question, retrieved_segments)

        retrieved_objects = [
            RetrievedObject(
                id=segment.id,
                label=segment.label,
                types=segment.types,
                description=segment.description,
                score=segment.score,
            )
            for segment in retrieved_segments
        ]
        relevant_entities = [
            RelevantEntity(
                id=obj.id,
                label=obj.label,
                types=obj.types,
                score=obj.score,
            )
            for obj in retrieved_objects
        ]
        evidence = self._evidence_from_segments(retrieved_segments)

        return ExplanationResponse(
            answer=self._attach_entity_citations(answer, relevant_entities),
            relevant_entities=relevant_entities,
            evidence=evidence,
            retrieved_objects=retrieved_objects,
            token_usage=self.last_token_usage,
        )

    def search(self, phrase: str, limit: Optional[int] = None) -> list[ObjectSegment]:
        normalized_phrase = phrase.strip()
        if not normalized_phrase:
            return []

        query_vector = self.embeddings_model.embed_query(normalized_phrase)
        scored_segments = [
            (_cosine_similarity(query_vector, segment.vector), segment)
            for segment in self.segments
            if segment.vector
        ]
        scored_segments.sort(key=lambda item: item[0], reverse=True)

        results: list[ObjectSegment] = []
        for score, segment in scored_segments[: limit or self.top_k]:
            segment_copy = ObjectSegment(
                iri=segment.iri,
                id=segment.id,
                label=segment.label,
                types=list(segment.types),
                description=segment.description,
                evidence=list(segment.evidence),
                score=score,
                vector=list(segment.vector),
            )
            results.append(segment_copy)
        return results

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

    def _build_object_segments(self) -> list[ObjectSegment]:
        subjects = sorted(
            {
                str(subject)
                for subject in self.kg.subjects()
                if isinstance(subject, URIRef)
            }
        )
        return [
            self._build_object_segment(subject_iri)
            for subject_iri in subjects
        ]

    def _build_object_segment(self, subject_iri: str) -> ObjectSegment:
        subject = URIRef(subject_iri)
        object_id = self.to_curie(subject_iri)
        label = self._entity_label(subject_iri)
        types: list[str] = []
        property_lines: list[str] = []
        evidence: list[EvidenceTriple] = []

        for predicate, obj in sorted(
            self.kg.predicate_objects(subject),
            key=lambda item: (str(item[0]), str(item[1])),
        ):
            predicate_iri = str(predicate)
            predicate_id = self.to_curie(predicate_iri)
            predicate_label = self._predicate_label(predicate_iri)
            object_is_literal = isinstance(obj, Literal)
            object_iri = str(obj) if isinstance(obj, URIRef) else None
            object_id_value = self.to_curie(object_iri) if object_iri else None
            object_label = (
                self._entity_label(object_iri)
                if object_iri
                else _clean_literal_text(str(obj))
            )

            if predicate_iri == str(RDF.type) and object_id_value:
                types.append(object_id_value)
                continue

            property_lines.append(
                f"{predicate_id} ({predicate_label}): "
                f"{object_id_value or json.dumps(object_label, ensure_ascii=False)}"
                + (f" [{object_label}]" if object_id_value else "")
            )
            evidence.append(
                EvidenceTriple(
                    subject_id=object_id,
                    subject_label=label,
                    predicate_id=predicate_id,
                    predicate_label=predicate_label,
                    object_id=object_id_value,
                    object_label=object_label,
                    object_is_literal=object_is_literal,
                    direction="outgoing",
                )
            )

        description_parts = [f"Object: {object_id} [{label}]."]
        if types:
            description_parts.append("Types: " + ", ".join(dict.fromkeys(types)) + ".")
        if property_lines:
            description_parts.append("Properties: " + "; ".join(property_lines) + ".")

        return ObjectSegment(
            iri=subject_iri,
            id=object_id,
            label=label,
            types=list(dict.fromkeys(types)),
            description=" ".join(description_parts),
            evidence=evidence,
        )

    def _embed_segments(self) -> None:
        if not self.segments:
            return
        descriptions = [segment.description for segment in self.segments]
        vectors = self.embeddings_model.embed_documents(descriptions)
        for segment, vector in zip(self.segments, vectors):
            segment.vector = list(vector)

    def _answer_with_llm(
        self,
        question: str,
        retrieved_segments: list[ObjectSegment],
    ) -> str:
        self.last_token_usage = _empty_token_usage()
        system_prompt = (
            "Answer workflow provenance questions using only the retrieved object "
            "descriptions. If the retrieved objects are insufficient, say so directly."
        )
        context = "\n\n".join(
            f"[{index}] similarity={segment.score:.4f}\n"
            f"{segment.description}"
            for index, segment in enumerate(retrieved_segments, start=1)
        )
        prompt = (
            "Retrieved object descriptions:\n"
            f"{context or '- none'}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Return JSON with one field:\n"
            "- answer: a concise answer grounded only in the retrieved object descriptions"
        )

        structured_answer = self._query_langchain_structured(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        if structured_answer:
            return structured_answer

        try:
            if hasattr(self.llm, "structured_generate"):
                response = _run_async(
                    self.llm.structured_generate(
                        prompt=prompt,
                        structure=LLMAnswerPayload,
                        system_prompt=system_prompt,
                    )
                )
                answer = LLMAnswerPayload.model_validate(response).answer.strip()
                if answer:
                    self.last_token_usage = self._token_usage_for_call(
                        system_prompt=system_prompt,
                        prompt=prompt,
                        completion_text=self._payload_text({"answer": answer}),
                        response=response,
                    )
                    return answer
        except Exception:
            pass

        raw_answer = self._query_langchain_raw(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        if raw_answer:
            return raw_answer

        try:
            if hasattr(self.llm, "generate"):
                raw = _run_async(
                    self.llm.generate(prompt=prompt, system_prompt=system_prompt)
                ).strip()
                if raw:
                    answer = self._parse_raw_answer(raw)
                    self.last_token_usage = self._token_usage_for_call(
                        system_prompt=system_prompt,
                        prompt=prompt,
                        completion_text=raw,
                    )
                    return answer
        except Exception:
            return ""

        return ""

    def _query_langchain_structured(
        self,
        *,
        prompt: str,
        system_prompt: str,
    ) -> str:
        chat_model = getattr(self.llm, "llm", None)
        if chat_model is None or not hasattr(chat_model, "with_structured_output"):
            return ""

        try:
            structured_llm = chat_model.with_structured_output(
                LLMAnswerPayload,
                method="function_calling",
                include_raw=True,
            )
            response = _run_async(
                structured_llm.ainvoke(self._langchain_messages(prompt, system_prompt))
            )
            if not isinstance(response, dict):
                return ""

            raw_response = response.get("raw")
            parsed = response.get("parsed")
            if parsed is None:
                parsing_error = response.get("parsing_error")
                if parsing_error:
                    raise parsing_error
                return ""

            answer = LLMAnswerPayload.model_validate(parsed).answer.strip()
            if answer:
                self.last_token_usage = self._token_usage_for_call(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    completion_text=self._payload_text({"answer": answer}),
                    response=raw_response,
                )
            return answer
        except Exception:
            return ""

    def _query_langchain_raw(
        self,
        *,
        prompt: str,
        system_prompt: str,
    ) -> str:
        chat_model = getattr(self.llm, "llm", None)
        if chat_model is None or not hasattr(chat_model, "ainvoke"):
            return ""

        try:
            response = _run_async(
                chat_model.ainvoke(self._langchain_messages(prompt, system_prompt))
            )
            raw = self._response_text(response).strip()
            if raw:
                answer = self._parse_raw_answer(raw)
                self.last_token_usage = self._token_usage_for_call(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    completion_text=raw,
                    response=response,
                )
                return answer
        except Exception:
            return ""
        return ""

    def _parse_raw_answer(self, raw_response: str) -> str:
        text = raw_response.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1).strip()
        try:
            data = json.loads(text)
            return LLMAnswerPayload.model_validate(data).answer.strip()
        except Exception:
            return raw_response.strip()

    def _fallback_answer(
        self,
        question: str,
        retrieved_segments: list[ObjectSegment],
    ) -> str:
        if not retrieved_segments:
            return "No similar KG objects were retrieved, so I could not answer the question."

        top_objects = ", ".join(
            f"{segment.label} ({segment.id})"
            for segment in retrieved_segments[: self.top_k]
        )
        return (
            f"I retrieved the most similar objects for the question '{question}': "
            f"{top_objects}. An LLM answer could not be generated from them."
        )

    def _evidence_from_segments(
        self,
        retrieved_segments: list[ObjectSegment],
    ) -> list[EvidenceTriple]:
        evidence: list[EvidenceTriple] = []
        seen: set[tuple[str, str, Optional[str], str]] = set()
        for segment in retrieved_segments:
            for triple in segment.evidence:
                key = (
                    triple.subject_id,
                    triple.predicate_id,
                    triple.object_id,
                    triple.object_label,
                )
                if key in seen:
                    continue
                seen.add(key)
                triple.score = segment.score
                evidence.append(triple)
                if len(evidence) >= self.max_evidence:
                    return evidence
        return evidence

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
                return f"{prefix}:{iri[len(namespace):]}"
        return iri

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
            candidates.extend([response.get("usage_metadata"), response.get("usage")])
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
            candidates.extend([metadata.get("token_usage"), metadata.get("usage"), metadata])

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
