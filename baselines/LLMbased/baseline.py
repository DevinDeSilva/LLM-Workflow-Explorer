from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

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
MODEL_PREDICATES = {
    "http://www.semanticweb.org/acer/ontologies/2026/1/WorkFlow/llm_model",
}
PROV_VALUE = "http://www.w3.org/ns/prov#value"
PROV_USED = "http://www.w3.org/ns/prov#used"
PROV_WAS_GENERATED_BY = "http://www.w3.org/ns/prov#wasGeneratedBy"
PROV_QUALIFIED_ASSOCIATION = "http://www.w3.org/ns/prov#qualifiedAssociation"
PROV_QUALIFIED_GENERATION = "http://www.w3.org/ns/prov#qualifiedGeneration"
PROV_HAD_PLAN = "http://www.w3.org/ns/prov#hadPlan"
PROVONE_HAD_ENTITY = "http://purl.dataone.org/provone/2015/01/15/ontology#hadEntity"
WORKFLOW_EXECUTES_TASK = "http://semanticscience.org/resource/SIO_000369"
WORKFLOW_LLM_INPUT = "http://semanticscience.org/resource/SIO_000230"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "does",
    "execution",
    "for",
    "from",
    "given",
    "happen",
    "happened",
    "handled",
    "how",
    "in",
    "is",
    "it",
    "method",
    "of",
    "program",
    "prompt",
    "request",
    "the",
    "their",
    "to",
    "used",
    "using",
    "was",
    "were",
    "what",
    "which",
    "who",
    "why",
}


def _unique(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        try:
            key: Any = item if hash(item) is not None else repr(item)
        except Exception:
            key = repr(item)
        if key in seen:
            continue
        seen.add(key)
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


def _tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    return {token for token in normalized.split() if token and token not in STOPWORDS}


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "relevant_entities": [asdict(entity) for entity in self.relevant_entities],
            "evidence": [asdict(triple) for triple in self.evidence],
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
    lexical_tokens: set[str] = field(default_factory=set)
    model_name: str = ""


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


@dataclass(slots=True)
class ExecutionRecord:
    iri: str
    id: str
    label: str
    program_iri: Optional[str]
    program_label: str
    used_entities: list[str]
    generated_entities: list[str]
    llm_entities: list[str]


class LLMRewritePayload(BaseModel):
    answer: str = Field(default="")


class GroundedWorkflowBaseline:
    """
    Grounding-first workflow explanation baseline.

    The baseline builds a local index over the KG and answers questions from
    retrieved entities and provenance relations. If an LLM is supplied, it is
    only used to rewrite the already-grounded summary.
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
    ) -> None:
        self.kg_path = Path(kg_path)
        self.ontology_path = Path(ontology_path)
        self.schema_json_path = Path(schema_json_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.application_description = application_description.strip()
        self.max_relevant_entities = max_relevant_entities
        self.max_evidence = max_evidence

        self.kg = Graph()
        self.kg.parse(self.kg_path, format="turtle")

        self.ontology = Graph()
        self.ontology.parse(self.ontology_path, format="turtle")

        self.namespaces = self._load_namespaces()
        self.schema = self._load_schema()
        self.entities = self._build_entity_index()
        self.predicates = self._build_predicate_index()
        self.triples = self._build_triples()
        self.outgoing_triples = self._build_triple_adjacency(direction="outgoing")
        self.incoming_triples = self._build_triple_adjacency(direction="incoming")
        self.execution_records = self._build_execution_records()

        self.llm = llm
        if self.llm is None and llm_config is not None:
            from src.llm import LLM

            self.llm = LLM(llm_type, llm_library, **llm_config)

    def request(self, user_query: str) -> dict[str, Any]:
        return self.answer(user_query).to_dict()

    def answer(self, user_query: str) -> ExplanationResponse:
        question = user_query.strip()
        if not question:
            return ExplanationResponse(
                answer="The question is empty, so there is no answer to return.",
                relevant_entities=[],
                evidence=[],
            )

        answer_text, relevant_entities, evidence = self._answer_grounded(question)
        if self.llm is not None:
            answer_text = self._rewrite_with_llm(
                question=question,
                answer_text=answer_text,
                relevant_entities=relevant_entities,
                evidence=evidence,
            )
        relevant_entities = relevant_entities[: self.max_relevant_entities]
        answer_text = self._attach_entity_citations(answer_text, relevant_entities)

        return ExplanationResponse(
            answer=answer_text,
            relevant_entities=relevant_entities,
            evidence=evidence[: self.max_evidence],
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

    def _answer_grounded(
        self, question: str
    ) -> tuple[str, list[RelevantEntity], list[EvidenceTriple]]:
        intent = self._detect_intent(question)

        if intent == "model":
            result = self._answer_model_question(question)
            if result is not None:
                return result

        if intent == "inputs":
            result = self._answer_inputs_question(question)
            if result is not None:
                return result

        result = self._answer_data_lineage_question(question)
        if result is not None:
            return result

        return self._answer_generic_question(question)

    def _answer_model_question(
        self, question: str
    ) -> tuple[str, list[RelevantEntity], list[EvidenceTriple]] | None:
        execution = self._find_best_execution(question)
        if "prompt" in _normalize_text(question):
            prompt_execution = self._best_prompt_execution()
            if prompt_execution is not None:
                execution = prompt_execution
        if execution is None:
            return None

        llm_entity = self._pick_execution_llm(execution)
        model_name = llm_entity.model_name if llm_entity else ""
        if not model_name:
            task_triple = self._find_first_triple(
                execution.iri,
                predicate_iri=WORKFLOW_EXECUTES_TASK,
            )
            if task_triple:
                match = re.search(r"using:\s*([A-Za-z0-9._-]+)", task_triple.object_label)
                if match:
                    model_name = match.group(1)

        if not model_name:
            return None

        input_labels = self._ordered_labels(execution.used_entities)
        input_phrase = ", ".join(input_labels[:2]) if input_labels else "the available inputs"
        answer = (
            f"The {execution.program_label} execution used the {model_name} model to handle "
            f"the prompt with inputs {input_phrase}."
        )

        relevant = self._build_relevant_entities(
            [
                execution.program_iri,
                execution.iri,
                llm_entity.iri if llm_entity else None,
                *execution.used_entities[:2],
            ]
        )
        evidence = self._build_evidence(
            [
                self._find_model_triple(llm_entity.iri) if llm_entity else None,
                *self._usage_triples(execution.iri),
            ]
        )
        return answer, relevant, evidence

    def _answer_inputs_question(
        self, question: str
    ) -> tuple[str, list[RelevantEntity], list[EvidenceTriple]] | None:
        execution = self._find_best_execution(question)
        if execution is None:
            return None

        input_labels = self._ordered_labels(execution.used_entities)
        if not input_labels:
            return None

        input_phrase = ", ".join(input_labels)
        output_labels = self._ordered_labels(execution.generated_entities)
        output_phrase = f" and produced {', '.join(output_labels)}" if output_labels else ""
        answer = (
            f"The {execution.program_label} execution ({execution.label}) ran with inputs "
            f"{input_phrase}{output_phrase}."
        )

        relevant = self._build_relevant_entities(
            [execution.iri, execution.program_iri, *execution.used_entities, *execution.generated_entities]
        )
        evidence = self._build_evidence(
            [
                *self._usage_triples(execution.iri),
                *self._generation_triples(execution.iri),
            ]
        )
        return answer, relevant, evidence

    def _answer_data_lineage_question(
        self, question: str
    ) -> tuple[str, list[RelevantEntity], list[EvidenceTriple]] | None:
        target_entity = self._find_best_entity(question)
        if target_entity is None:
            return None

        generation_triples = self._outgoing_matching(target_entity.iri, PROV_WAS_GENERATED_BY)
        usage_triples = self._incoming_matching(target_entity.iri, PROV_USED)

        if not generation_triples and not usage_triples:
            return None

        primary_generation = self._prefer_execution_with_program(generation_triples)
        generated_by_execution = (
            self.execution_records.get(primary_generation.object_iri) if primary_generation and primary_generation.object_iri else None
        )

        usage_execution_records = [
            self.execution_records[triple.subject_iri]
            for triple in usage_triples
            if triple.subject_iri in self.execution_records
        ]
        usage_execution_records = _unique(usage_execution_records)

        answer_parts = [f"The {target_entity.label}"]
        if generated_by_execution is not None:
            answer_parts.append(
                f"was produced by the {generated_by_execution.program_label} execution"
            )
        elif primary_generation is not None:
            answer_parts.append(
                f"was generated by {primary_generation.object_label}"
            )

        if usage_execution_records:
            used_by_labels = ", ".join(record.program_label for record in usage_execution_records[:3])
            connector = " and then" if len(answer_parts) > 1 else " was"
            answer_parts.append(f"{connector} used by {used_by_labels}")

        generated_outputs: list[str] = []
        for record in usage_execution_records[:1]:
            generated_outputs.extend(
                label
                for label in self._ordered_labels(record.generated_entities)
                if label.lower() != target_entity.label.lower()
            )
        if generated_outputs:
            answer_parts.append(f"to produce {', '.join(generated_outputs[:4])}")

        answer = " ".join(part.strip() for part in answer_parts if part.strip()) + "."

        relevant = self._build_relevant_entities(
            [
                target_entity.iri,
                generated_by_execution.program_iri if generated_by_execution else None,
                generated_by_execution.iri if generated_by_execution else None,
                *[record.program_iri for record in usage_execution_records],
                *[record.iri for record in usage_execution_records],
            ]
        )
        evidence = self._build_evidence([primary_generation, *usage_triples])
        return answer, relevant, evidence

    def _answer_generic_question(
        self, question: str
    ) -> tuple[str, list[RelevantEntity], list[EvidenceTriple]]:
        entity = self._find_best_entity(question)
        execution = self._find_best_execution(question)

        if execution is not None:
            input_labels = self._ordered_labels(execution.used_entities)
            output_labels = self._ordered_labels(execution.generated_entities)
            summary_bits = [f"{execution.program_label} ({execution.label})"]
            if input_labels:
                summary_bits.append(f"used {', '.join(input_labels)}")
            if output_labels:
                summary_bits.append(f"generated {', '.join(output_labels)}")
            answer = "The closest grounded workflow step is " + "; ".join(summary_bits) + "."
            relevant = self._build_relevant_entities(
                [execution.iri, execution.program_iri, *execution.used_entities, *execution.generated_entities]
            )
            evidence = self._build_evidence(
                [
                    *self._usage_triples(execution.iri),
                    *self._generation_triples(execution.iri),
                ]
            )
            return answer, relevant, evidence

        if entity is not None:
            triple_pool = self.outgoing_triples.get(entity.iri, [])[:3] + self.incoming_triples.get(entity.iri, [])[:3]
            evidence = self._build_evidence(triple_pool)
            answer = f"The closest grounded entity is {entity.label}."
            relevant = self._build_relevant_entities([entity.iri])
            return answer, relevant, evidence

        return (
            "I could not find enough grounded workflow evidence in the knowledge graph to answer that question.",
            [],
            [],
        )

    def _rewrite_with_llm(
        self,
        *,
        question: str,
        answer_text: str,
        relevant_entities: list[RelevantEntity],
        evidence: list[EvidenceTriple],
    ) -> str:
        entity_lines = "\n".join(
            f"- {entity.id} [{entity.label}]"
            for entity in relevant_entities
        )
        evidence_lines = "\n".join(
            f"- {triple.subject_id} [{triple.subject_label}] {triple.predicate_id} "
            f"[{triple.predicate_label}] {triple.object_id or json.dumps(triple.object_label, ensure_ascii=False)} "
            f"[{triple.object_label}]"
            for triple in evidence
        )
        prompt = (
            "Rewrite the grounded answer so it reads naturally but stays fully faithful to the evidence.\n\n"
            f"Question:\n{question}\n\n"
            f"Grounded draft answer:\n{answer_text}\n\n"
            f"Relevant entities:\n{entity_lines or '- none'}\n\n"
            f"Evidence triples:\n{evidence_lines or '- none'}\n\n"
            "Return JSON with one field:\n"
            "- answer: a concise natural-language answer grounded only in the evidence"
        )
        system_prompt = (
            "You rewrite grounded provenance answers. Do not add facts beyond the supplied evidence."
        )

        try:
            if hasattr(self.llm, "structured_generate"):
                response = _run_async(
                    self.llm.structured_generate(
                        prompt=prompt,
                        structure=LLMRewritePayload,
                        system_prompt=system_prompt,
                    )
                )
                rewritten = LLMRewritePayload.model_validate(response).answer.strip()
                if rewritten:
                    return rewritten
            if hasattr(self.llm, "generate"):
                raw = _run_async(self.llm.generate(prompt=prompt, system_prompt=system_prompt)).strip()
                if raw:
                    return raw
        except Exception:
            return answer_text

        return answer_text

    def _detect_intent(self, question: str) -> str:
        normalized = _normalize_text(question)
        if "model" in normalized and any(term in normalized for term in {"handle", "handled", "using", "used"}):
            return "model"
        if "input" in normalized or "inputs" in normalized:
            return "inputs"
        if any(term in normalized for term in {"what happened", "happened to", "lineage", "trace"}):
            return "lineage"
        return "generic"

    def _find_best_execution(self, question: str) -> ExecutionRecord | None:
        entity = self._find_best_entity(question, prefer_execution_related=True)
        if entity is not None:
            if entity.iri in self.execution_records:
                return self.execution_records[entity.iri]
            execution = self._execution_from_neighbor(entity.iri)
            if execution is not None:
                return execution

        scored: list[tuple[float, ExecutionRecord]] = []
        question_tokens = _tokens(question)
        for record in self.execution_records.values():
            score = 0.0
            label_tokens = _tokens(record.label) | _tokens(record.program_label)
            overlap = question_tokens & label_tokens
            if overlap:
                score += 2.5 * len(overlap)
            if "llm" in question_tokens and "llm" in label_tokens:
                score += 1.5
            if "chat" in question_tokens and "chat" in label_tokens:
                score += 1.5
            if "prompt" in question_tokens:
                input_labels = " ".join(self._ordered_labels(record.used_entities))
                if "prompt" in _normalize_text(input_labels):
                    score += 1.5
                if self._execution_has_prompt_inputs(record):
                    score += 2.5
            if score > 0:
                scored.append((score, record))

        if not scored:
            return None

        scored.sort(key=lambda item: (item[0], item[1].program_label == "LLM Chat"), reverse=True)
        return scored[0][1]

    def _find_best_entity(
        self, question: str, *, prefer_execution_related: bool = False
    ) -> EntityRecord | None:
        question_tokens = _tokens(question)
        normalized_question = _normalize_text(question)
        best_score = 0.0
        best_record: EntityRecord | None = None

        for record in self.entities.values():
            score = 0.0
            overlap = question_tokens & record.lexical_tokens
            if overlap:
                score += 2.0 * len(overlap)

            if record.normalized_label and record.normalized_label in normalized_question:
                score += 3.0

            if any(alias and _normalize_text(alias) in normalized_question for alias in record.aliases):
                score += 2.0

            if "generated answer" in normalized_question and record.normalized_label == "generated answer":
                score += 4.0
            if "user prompt" in normalized_question and record.normalized_label == "user prompt":
                score += 4.0
            if "system prompt" in normalized_question and record.normalized_label == "system prompt":
                score += 4.0
            if "llm chat" in normalized_question and record.normalized_label == "llm chat":
                score += 4.0

            if prefer_execution_related and self._execution_from_neighbor(record.iri) is not None:
                score += 0.5

            if score > best_score:
                best_score = score
                best_record = record

        return best_record if best_score > 0 else None

    def _execution_from_neighbor(self, entity_iri: str) -> ExecutionRecord | None:
        if entity_iri in self.execution_records:
            return self.execution_records[entity_iri]

        for triple in self.incoming_triples.get(entity_iri, []):
            if triple.subject_iri in self.execution_records and triple.predicate_iri in {PROV_USED}:
                return self.execution_records[triple.subject_iri]
        for triple in self.outgoing_triples.get(entity_iri, []):
            if triple.object_iri in self.execution_records and triple.predicate_iri == PROV_WAS_GENERATED_BY:
                return self.execution_records[triple.object_iri]
        for execution in self.execution_records.values():
            if entity_iri == execution.program_iri or entity_iri in execution.llm_entities:
                return execution
        return None

    def _pick_execution_llm(self, execution: ExecutionRecord) -> EntityRecord | None:
        for llm_iri in execution.llm_entities:
            if llm_iri in self.entities:
                return self.entities[llm_iri]
        return None

    def _ordered_labels(self, entity_iris: list[str]) -> list[str]:
        labels = [
            self.entities[iri].label
            for iri in entity_iris
            if iri in self.entities and self.entities[iri].label
        ]
        if not labels:
            return []

        priority = {
            "user prompt": 0,
            "system prompt": 1,
            "generated answer": 2,
        }
        return sorted(
            _unique(labels),
            key=lambda label: (priority.get(label.lower(), 50), label.lower()),
        )

    def _execution_has_prompt_inputs(self, execution: ExecutionRecord) -> bool:
        labels = {label.lower() for label in self._ordered_labels(execution.used_entities)}
        return {"user prompt", "system prompt"} <= labels

    def _best_prompt_execution(self) -> ExecutionRecord | None:
        candidates = [
            execution
            for execution in self.execution_records.values()
            if self._execution_has_prompt_inputs(execution)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda execution: execution.program_label == "LLM Chat", reverse=True)
        return candidates[0]

    def _find_model_triple(self, entity_iri: str) -> TripleRecord | None:
        return self._find_first_triple(entity_iri, predicate_iri=next(iter(MODEL_PREDICATES)))

    def _usage_triples(self, execution_iri: str) -> list[TripleRecord]:
        return [
            triple
            for triple in self.outgoing_triples.get(execution_iri, [])
            if triple.predicate_iri == PROV_USED
        ]

    def _generation_triples(self, execution_iri: str) -> list[TripleRecord]:
        triples: list[TripleRecord] = []
        for candidate in self.triples:
            if candidate.predicate_iri == PROV_WAS_GENERATED_BY and candidate.object_iri == execution_iri:
                triples.append(candidate)
        return triples

    def _incoming_matching(self, entity_iri: str, predicate_iri: str) -> list[TripleRecord]:
        return [
            triple
            for triple in self.incoming_triples.get(entity_iri, [])
            if triple.predicate_iri == predicate_iri
        ]

    def _outgoing_matching(self, entity_iri: str, predicate_iri: str) -> list[TripleRecord]:
        return [
            triple
            for triple in self.outgoing_triples.get(entity_iri, [])
            if triple.predicate_iri == predicate_iri
        ]

    def _prefer_execution_with_program(self, triples: list[TripleRecord]) -> TripleRecord | None:
        if not triples:
            return None

        def sort_key(triple: TripleRecord) -> tuple[int, int]:
            execution = self.execution_records.get(triple.object_iri or "")
            if execution is None:
                return (0, 0)
            if execution.program_label == "LLM Chat":
                return (3, 0)
            if execution.program_label:
                return (2, 0)
            return (1, 0)

        return max(triples, key=sort_key)

    def _build_relevant_entities(self, entity_iris: list[Optional[str]]) -> list[RelevantEntity]:
        relevant: list[RelevantEntity] = []
        seen: set[str] = set()

        for index, iri in enumerate(entity_iris):
            if not iri or iri not in self.entities:
                continue
            record = self.entities[iri]
            if record.curie in seen:
                continue
            seen.add(record.curie)
            relevant.append(
                RelevantEntity(
                    id=record.curie,
                    label=record.label,
                    types=record.types,
                    score=max(0.0, 1.0 - (index * 0.1)),
                )
            )
        return relevant

    def _build_evidence(self, triples: list[TripleRecord | None]) -> list[EvidenceTriple]:
        evidence: list[EvidenceTriple] = []
        seen: set[tuple[str, str, Optional[str], str]] = set()

        for index, triple in enumerate(triple for triple in triples if triple is not None):
            key = (triple.subject_id, triple.predicate_id, triple.object_id, triple.object_label)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                EvidenceTriple(
                    subject_id=triple.subject_id,
                    subject_label=triple.subject_label,
                    predicate_id=triple.predicate_id,
                    predicate_label=triple.predicate_label,
                    object_id=triple.object_id,
                    object_label=triple.object_label,
                    object_is_literal=triple.object_is_literal,
                    direction="outgoing",
                    score=max(0.0, 1.0 - (index * 0.1)),
                )
            )
        return evidence

    def _find_first_triple(self, subject_iri: str, *, predicate_iri: str) -> TripleRecord | None:
        for triple in self.outgoing_triples.get(subject_iri, []):
            if triple.predicate_iri == predicate_iri:
                return triple
        return None

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
                    elif predicate_iri in MODEL_PREDICATES:
                        subject_record.model_name = cleaned_literal

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
            record.lexical_tokens = _tokens(
                " ".join(
                    [
                        record.label,
                        record.curie,
                        *record.aliases,
                        *record.descriptions[:1],
                        *record.identifiers,
                    ]
                )
            )
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

    def _build_triple_adjacency(self, *, direction: str) -> dict[str, list[TripleRecord]]:
        adjacency: dict[str, list[TripleRecord]] = defaultdict(list)
        if direction == "outgoing":
            for triple in self.triples:
                adjacency[triple.subject_iri].append(triple)
            return adjacency

        for triple in self.triples:
            if triple.object_iri:
                adjacency[triple.object_iri].append(triple)
        return adjacency

    def _build_execution_records(self) -> dict[str, ExecutionRecord]:
        executions: dict[str, ExecutionRecord] = {}
        execution_type_iri = self.expand_curie("provone:Execution")
        for iri, record in self.entities.items():
            if self.expand_curie("provone:Execution") not in {self.expand_curie(t) for t in record.types}:
                continue

            program_iri: Optional[str] = None
            program_label = ""
            for assoc in self.kg.objects(URIRef(iri), URIRef(PROV_QUALIFIED_ASSOCIATION)):
                for plan in self.kg.objects(assoc, URIRef(PROV_HAD_PLAN)):
                    program_iri = str(plan)
                    if program_iri in self.entities:
                        program_label = self.entities[program_iri].label
                        break
                if program_iri:
                    break

            used_entities = [
                str(obj)
                for obj in self.kg.objects(URIRef(iri), URIRef(PROV_USED))
                if isinstance(obj, URIRef)
            ]
            generated_entities = [
                triple.subject_iri
                for triple in self.triples
                if triple.predicate_iri == PROV_WAS_GENERATED_BY and triple.object_iri == iri
            ]
            llm_entities = self._llm_entities_for_execution(iri)
            executions[iri] = ExecutionRecord(
                iri=iri,
                id=record.curie,
                label=record.label,
                program_iri=program_iri,
                program_label=program_label or record.label,
                used_entities=used_entities,
                generated_entities=generated_entities,
                llm_entities=llm_entities,
            )

        return executions

    def _llm_entities_for_execution(self, execution_iri: str) -> list[str]:
        execution_key = _local_name(execution_iri)
        llm_candidates: list[str] = []
        for iri, record in self.entities.items():
            if not any(entity_type.startswith("workflow:Large_Language_Models") for entity_type in record.types):
                continue
            local_name = _local_name(iri)
            if execution_key and execution_key in local_name:
                llm_candidates.append(iri)

        if llm_candidates:
            return _unique(llm_candidates)

        used_inputs = {
            triple.object_iri
            for triple in self.outgoing_triples.get(execution_iri, [])
            if triple.predicate_iri == PROV_USED and triple.object_iri
        }
        for iri, record in self.entities.items():
            if not record.model_name:
                continue
            llm_inputs = {
                triple.object_iri
                for triple in self.outgoing_triples.get(iri, [])
                if triple.predicate_iri == WORKFLOW_LLM_INPUT and triple.object_iri
            }
            if used_inputs and used_inputs <= llm_inputs:
                llm_candidates.append(iri)
        return _unique(llm_candidates)

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
        if record.model_name:
            return f"{record.model_name} model"

        if record.aliases:
            alias = record.aliases[0]
            if " " not in alias and any(char in alias for char in {"_", "-"}):
                humanized_alias = _humanize_identifier(alias)
                if humanized_alias:
                    return humanized_alias
            return alias

        if record.identifiers and any(entity_type == "provone:Execution" for entity_type in record.types):
            return f"execution {record.identifiers[0]}"

        local_name = _local_name(record.iri)
        cleaned_local_name = re.sub(r"^[A-Za-z_]+-id_[0-9_]+-", "", local_name)
        if ":" in cleaned_local_name:
            cleaned_local_name = cleaned_local_name.split(":", 1)[1]
        humanized = _humanize_identifier(cleaned_local_name)
        return humanized or record.curie

    def _first_literal(self, graph: Graph, subject_iri: str, predicate_iris: set[str]) -> str:
        subject = URIRef(subject_iri)
        for predicate_iri in predicate_iris:
            for literal in graph.objects(subject, URIRef(predicate_iri)):
                if isinstance(literal, Literal):
                    cleaned = _clean_literal_text(str(literal))
                    if cleaned:
                        return cleaned
        return ""
