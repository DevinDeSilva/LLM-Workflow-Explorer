from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

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
IMPORTANT_PREDICATES = {
    "prov:used",
    "prov:wasGeneratedBy",
    "prov:wasInformedBy",
    "prov:qualifiedAssociation",
    "prov:qualifiedUsage",
    "prov:qualifiedGeneration",
    "prov:wasAssociatedWith",
    "prov:hadPlan",
    "provone:hadEntity",
    "provone:hadMember",
    "provone:hadInPort",
    "provone:hadOutPort",
    "provone:hasInPort",
    "provone:hasOutPort",
    "sio:SIO_000229",
    "sio:SIO_000230",
    "sio:SIO_000232",
    "sio:SIO_000313",
    "sio:SIO_000369",
    "workflow:llm_model",
}
STRUCTURAL_TYPES = {
    "prov:Association",
    "prov:Generation",
    "prov:Usage",
    "provone:Port",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
INTENT_KEYWORDS: dict[str, set[str]] = {
    "model": {"model"},
    "program": {"program", "plan", "function", "step"},
    "execution": {"execution", "run", "ran"},
    "input": {"input", "inputs", "used", "use", "prompt", "context"},
    "output": {"output", "outputs", "generated", "generate", "produced", "result", "answer"},
    "agent": {"agent", "user", "responsible", "who"},
    "time": {"time", "when", "start", "started", "end", "ended"},
    "explanation": {"happened", "explain", "summary", "trace", "workflow"},
}
TYPE_INTENT_BONUS: dict[str, dict[str, float]] = {
    "model": {
        "workflow:Large_Language_Models": 5.0,
        "workflow:Large_Language_Model_Output": 3.0,
        "workflow:Generative_Task": 2.0,
    },
    "program": {
        "provone:Program": 5.0,
        "provone:Execution": 2.0,
    },
    "execution": {
        "provone:Execution": 5.0,
        "workflow:Generative_Task": 2.5,
    },
    "input": {
        "provone:Data": 2.5,
        "provone:Collection": 2.5,
        "provone:Port": 2.0,
        "provone:Execution": 1.0,
    },
    "output": {
        "provone:Data": 3.0,
        "provone:Collection": 3.0,
        "workflow:Large_Language_Model_Output": 3.0,
        "provone:Execution": 1.0,
    },
    "agent": {
        "provone:User": 4.0,
        "prov:Agent": 4.0,
        "provone:Execution": 1.0,
    },
    "time": {
        "provone:Execution": 3.0,
        "workflow:Large_Language_Models": 1.5,
    },
    "explanation": {
        "provone:Execution": 3.0,
        "provone:Program": 2.0,
        "provone:Data": 2.0,
        "workflow:Large_Language_Models": 2.0,
    },
}


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
    literal_facts: list[tuple[str, str]] = field(default_factory=list)
    normalized_label: str = ""
    label_tokens: set[str] = field(default_factory=set)
    search_tokens: set[str] = field(default_factory=set)
    search_text: str = ""


class GroundedWorkflowBaseline:
    """
    A deterministic baseline for workflow explanation over a provenance KG.

    The baseline loads the execution KG together with the workflow ontology and
    returns grounded natural-language answers plus the KG entities used.
    """

    def __init__(
        self,
        kg_path: str | Path,
        ontology_path: str | Path,
        schema_json_path: str | Path,
        metadata_path: str | Path | None = None,
        *,
        max_hops: int = 2,
        max_grounded_entities: int = 6,
        max_evidence: int = 8,
    ) -> None:
        self.kg_path = Path(kg_path)
        self.ontology_path = Path(ontology_path)
        self.schema_json_path = Path(schema_json_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.max_hops = max_hops
        self.max_grounded_entities = max_grounded_entities
        self.max_evidence = max_evidence

        self.kg = Graph()
        self.kg.parse(self.kg_path, format="turtle")

        self.ontology = Graph()
        self.ontology.parse(self.ontology_path, format="turtle")

        self.namespaces = self._load_namespaces()
        self.schema = self._load_schema()
        self.schema_metadata = self._build_schema_metadata()
        self.entities = self._build_entity_index()

    def request(self, user_query: str) -> dict[str, Any]:
        response = self.answer(user_query)
        return response.to_dict()

    def answer(self, user_query: str) -> ExplanationResponse:
        query = user_query.strip()
        if not query:
            return ExplanationResponse(
                answer="The question is empty, so there is no grounded explanation to return.",
                relevant_entities=[],
                evidence=[],
            )

        query_tokens = set(_tokenize(query))
        intents = self._detect_intents(query_tokens)
        anchors = self._ground_entities(query, query_tokens, intents)

        if not anchors:
            return ExplanationResponse(
                answer=(
                    "I could not ground the question to any entity in the knowledge graph. "
                    "Try mentioning a workflow step, execution, prompt, output, or model name."
                ),
                relevant_entities=[],
                evidence=[],
            )

        entity_scores, evidence = self._collect_neighborhood(anchors, query_tokens, intents)
        relevant_entities = self._build_relevant_entities(entity_scores, intents)
        answer = self._compose_answer(query, relevant_entities, evidence, intents)
        return ExplanationResponse(
            answer=answer,
            relevant_entities=relevant_entities,
            evidence=evidence[: self.max_evidence],
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

    def _load_schema(self) -> dict[str, Any]:
        return json.loads(self.schema_json_path.read_text(encoding="utf-8"))

    def _build_schema_metadata(self) -> dict[str, dict[str, Any]]:
        metadata: dict[str, dict[str, Any]] = {}
        terms = list(self.schema.get("classes", []))
        terms.extend(self.schema.get("object_properties", {}).keys())

        for term in terms:
            iri = self.expand_curie(term)
            label = self._first_literal(self.ontology, iri, LABEL_PREDICATES)
            description = self._first_literal(self.ontology, iri, DESCRIPTION_PREDICATES)
            metadata[term] = {
                "iri": iri,
                "label": label or _humanize_identifier(_local_name(iri)),
                "description": description or "",
                "connections": self.schema.get("object_properties", {}).get(term, {}).get("connections", []),
            }
        return metadata

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
                predicate_curie = self.to_curie(predicate_iri)

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
                    else:
                        subject_record.literal_facts.append((predicate_curie, cleaned_literal))

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
            record.literal_facts = _unique(record.literal_facts)
            record.label = self._derive_entity_label(record)
            record.normalized_label = _normalize_text(record.label)

            search_parts = [record.label, record.curie]
            search_parts.extend(record.aliases)
            search_parts.extend(record.descriptions[:2])
            search_parts.extend(record.identifiers)
            search_parts.extend(record.types)
            search_parts.extend(
                text for predicate, text in record.literal_facts if predicate in IMPORTANT_PREDICATES or len(text) < 200
            )

            record.label_tokens = set(_tokenize(record.label))
            record.search_tokens = set()
            for part in search_parts:
                record.search_tokens.update(_tokenize(part))
            record.search_text = " ".join(_normalize_text(part) for part in search_parts if part)
        return entities

    def _derive_entity_label(self, record: EntityRecord) -> str:
        if record.aliases:
            return record.aliases[0]
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

    def _detect_intents(self, query_tokens: set[str]) -> set[str]:
        intents = {
            intent
            for intent, keywords in INTENT_KEYWORDS.items()
            if query_tokens.intersection(keywords)
        }
        return intents or {"explanation"}

    def _ground_entities(
        self,
        query: str,
        query_tokens: set[str],
        intents: set[str],
    ) -> list[tuple[str, float]]:
        normalized_query = _normalize_text(query)
        scored_entities: list[tuple[str, float]] = []
        for iri, record in self.entities.items():
            score = self._score_entity(record, normalized_query, query_tokens, intents)
            if score > 0:
                scored_entities.append((iri, score))
        scored_entities.sort(key=lambda item: item[1], reverse=True)
        return scored_entities[: self.max_grounded_entities]

    def _score_entity(
        self,
        record: EntityRecord,
        normalized_query: str,
        query_tokens: set[str],
        intents: set[str],
    ) -> float:
        score = 0.0
        overlap = query_tokens.intersection(record.search_tokens)
        score += 2.0 * len(overlap)

        label_overlap = query_tokens.intersection(record.label_tokens)
        score += 1.5 * len(label_overlap)

        if record.normalized_label and record.normalized_label in normalized_query:
            score += 6.0

        if record.curie.lower() in normalized_query:
            score += 6.0

        for predicate, literal_value in record.literal_facts[:8]:
            literal_norm = _normalize_text(literal_value)
            if literal_norm and literal_norm in normalized_query and len(literal_norm) > 8:
                score += 3.0
            literal_tokens = set(_tokenize(literal_value))
            score += 0.3 * len(query_tokens.intersection(literal_tokens))
            if predicate == "workflow:llm_model" and query_tokens.intersection({"model", "llm"}):
                score += 4.0

        for intent in intents:
            for entity_type in record.types:
                score += TYPE_INTENT_BONUS.get(intent, {}).get(entity_type, 0.0)

        if "provone:Execution" in record.types:
            program_iri = self._program_for_execution(record.iri)
            if program_iri and program_iri in self.entities:
                program_label = self.entities[program_iri].normalized_label
                if program_label and program_label in normalized_query:
                    score += 6.0

        if "workflow:Large_Language_Models" in record.types:
            connected_execution_labels = [
                self.entities[program_iri].normalized_label
                for execution_iri in self._executions_for_llm(record.iri)
                for program_iri in [self._program_for_execution(execution_iri)]
                if program_iri and program_iri in self.entities
            ]
            for execution_label in connected_execution_labels:
                if execution_label and execution_label in normalized_query:
                    score += 5.0

            if "prompt" in query_tokens:
                input_labels = [
                    self.entities[data_iri].normalized_label
                    for data_iri in self._objects(record.iri, "sio:SIO_000230")
                    if data_iri in self.entities
                ]
                if any("user prompt" in label for label in input_labels):
                    score += 3.0
                if any("system prompt" in label for label in input_labels):
                    score += 2.0

        if any(entity_type in STRUCTURAL_TYPES for entity_type in record.types):
            score -= 0.6

        return score

    def _collect_neighborhood(
        self,
        anchors: Sequence[tuple[str, float]],
        query_tokens: set[str],
        intents: set[str],
    ) -> tuple[dict[str, float], list[EvidenceTriple]]:
        entity_scores = {iri: score for iri, score in anchors}
        evidence_by_key: dict[tuple[str, str, Optional[str], str], EvidenceTriple] = {}
        queue = deque((iri, 0, score) for iri, score in anchors)
        seen_best_depth = {iri: 0 for iri, _ in anchors}
        anchor_ids = {iri for iri, _ in anchors}

        while queue:
            iri, depth, path_score = queue.popleft()
            for evidence in self._iter_evidence_for_entity(iri):
                evidence.score = self._score_evidence(evidence, query_tokens, intents, anchor_ids)
                if evidence.score <= 0:
                    continue

                evidence_key = (
                    evidence.subject_id,
                    evidence.predicate_id,
                    evidence.object_id,
                    evidence.direction,
                )
                previous = evidence_by_key.get(evidence_key)
                if previous is None or evidence.score > previous.score:
                    evidence_by_key[evidence_key] = evidence

                next_entity_id = evidence.object_id if evidence.direction == "outgoing" else evidence.subject_id
                if evidence.object_is_literal or not next_entity_id:
                    continue

                next_iri = self.expand_curie(next_entity_id)
                next_score = path_score * 0.45 + evidence.score
                if next_score > entity_scores.get(next_iri, 0):
                    entity_scores[next_iri] = next_score

                if depth + 1 > self.max_hops:
                    continue
                best_depth = seen_best_depth.get(next_iri)
                if best_depth is None or depth + 1 < best_depth or next_score > entity_scores.get(next_iri, 0):
                    seen_best_depth[next_iri] = depth + 1
                    queue.append((next_iri, depth + 1, next_score))

        ranked_evidence = sorted(evidence_by_key.values(), key=lambda item: item.score, reverse=True)
        return entity_scores, ranked_evidence[: self.max_evidence]

    def _iter_evidence_for_entity(self, iri: str) -> Iterable[EvidenceTriple]:
        node = URIRef(iri)
        subject_record = self.entities.get(iri)
        if subject_record is None:
            return []

        for predicate, obj in self.kg.predicate_objects(node):
            predicate_curie = self.to_curie(str(predicate))
            yield EvidenceTriple(
                subject_id=subject_record.curie,
                subject_label=subject_record.label,
                predicate_id=predicate_curie,
                predicate_label=self._predicate_label(predicate_curie),
                object_id=self.to_curie(str(obj)) if isinstance(obj, URIRef) else None,
                object_label=self._object_label(obj),
                object_is_literal=not isinstance(obj, URIRef),
                direction="outgoing",
            )

        for subject, predicate in self.kg.subject_predicates(node):
            other_record = self.entities.get(str(subject))
            if other_record is None:
                continue
            predicate_curie = self.to_curie(str(predicate))
            yield EvidenceTriple(
                subject_id=other_record.curie,
                subject_label=other_record.label,
                predicate_id=predicate_curie,
                predicate_label=self._predicate_label(predicate_curie),
                object_id=subject_record.curie,
                object_label=subject_record.label,
                object_is_literal=False,
                direction="incoming",
            )

    def _score_evidence(
        self,
        evidence: EvidenceTriple,
        query_tokens: set[str],
        intents: set[str],
        anchor_ids: set[str],
    ) -> float:
        score = 0.2
        predicate_tokens = set(_tokenize(evidence.predicate_label))
        subject_tokens = set(_tokenize(evidence.subject_label))
        object_tokens = set(_tokenize(evidence.object_label))

        score += 1.2 * len(query_tokens.intersection(predicate_tokens))
        score += 1.0 * len(query_tokens.intersection(subject_tokens))
        score += 1.0 * len(query_tokens.intersection(object_tokens))

        if evidence.predicate_id in IMPORTANT_PREDICATES:
            score += 1.2

        if evidence.subject_id in anchor_ids or evidence.object_id in anchor_ids:
            score += 1.5

        if evidence.predicate_id == "workflow:llm_model" and "model" in intents:
            score += 4.0
        if evidence.predicate_id in {"prov:used", "provone:hadEntity"} and "input" in intents:
            score += 2.0
        if evidence.predicate_id in {"prov:qualifiedGeneration", "prov:wasGeneratedBy"} and "output" in intents:
            score += 2.0
        if evidence.predicate_id in {"prov:wasAssociatedWith", "prov:agent"} and "agent" in intents:
            score += 2.0
        if evidence.predicate_id in {"prov:startedAt", "prov:endedAt", "prov:atTime"} and "time" in intents:
            score += 2.0

        return score

    def _build_relevant_entities(
        self,
        entity_scores: dict[str, float],
        intents: set[str],
    ) -> list[RelevantEntity]:
        ranked_entities = sorted(entity_scores.items(), key=lambda item: item[1], reverse=True)
        results: list[RelevantEntity] = []
        seen_labels: set[str] = set()
        for iri, score in ranked_entities:
            record = self.entities.get(iri)
            if record is None:
                continue
            if not record.types and not record.aliases and not record.literal_facts:
                continue
            adjusted_score = score
            if any(entity_type in STRUCTURAL_TYPES for entity_type in record.types):
                adjusted_score -= 1.0
            if adjusted_score <= 0:
                continue

            normalized_label = record.normalized_label or record.curie.lower()
            if normalized_label in seen_labels:
                continue
            seen_labels.add(normalized_label)

            results.append(
                RelevantEntity(
                    id=record.curie,
                    label=record.label,
                    types=record.types,
                    score=round(adjusted_score, 2),
                )
            )
            if len(results) >= self.max_grounded_entities:
                break

        results.sort(
            key=lambda entity: self._entity_priority(entity, intents),
            reverse=True,
        )
        return results[: self.max_grounded_entities]

    def _entity_priority(self, entity: RelevantEntity, intents: set[str]) -> float:
        priority = entity.score
        for intent in intents:
            for entity_type in entity.types:
                priority += TYPE_INTENT_BONUS.get(intent, {}).get(entity_type, 0.0)
        if any(entity_type in STRUCTURAL_TYPES for entity_type in entity.types):
            priority -= 0.75
        return priority

    def _compose_answer(
        self,
        query: str,
        relevant_entities: list[RelevantEntity],
        evidence: list[EvidenceTriple],
        intents: set[str],
    ) -> str:
        answer_parts: list[str] = []
        primary_entity = self._select_focus_entity(relevant_entities, intents)
        if primary_entity:
            primary_iri = self.expand_curie(primary_entity.id)
            primary_description = self._describe_entity(primary_iri, intents)
            if primary_description:
                answer_parts.append(primary_description)

            if "output" in intents or "explanation" in intents:
                support_iri = self._support_entity_for(primary_iri, relevant_entities, {"provone:Execution", "workflow:Large_Language_Models"})
                if support_iri and support_iri != primary_iri:
                    support_description = self._describe_entity(support_iri, intents)
                    if support_description and support_description not in answer_parts:
                        answer_parts.append(support_description)
            elif "model" in intents:
                support_iri = self._support_entity_for(primary_iri, relevant_entities, {"provone:Execution"})
                if support_iri and support_iri != primary_iri:
                    support_description = self._describe_entity(support_iri, {"execution"})
                    if support_description and support_description not in answer_parts:
                        answer_parts.append(support_description)

        if not answer_parts:
            for entity in relevant_entities:
                iri = self.expand_curie(entity.id)
                description = self._describe_entity(iri, intents)
                if description:
                    answer_parts.append(description)
                    break

        if not answer_parts and evidence:
            answer_parts.append(self._fallback_from_evidence(evidence[:3]))

        if not answer_parts:
            answer_parts.append(
                "The graph contains related entities, but I could not assemble a grounded explanation from the matched neighborhood."
            )

        answer = " ".join(answer_parts)
        if relevant_entities:
            entity_names = ", ".join(entity.label for entity in relevant_entities[:4])
            answer = f"{answer} Relevant KG entities: {entity_names}."
        return answer

    def _select_focus_entity(
        self,
        relevant_entities: Sequence[RelevantEntity],
        intents: set[str],
    ) -> Optional[RelevantEntity]:
        type_priority = {
            "model": ["workflow:Large_Language_Models", "provone:Execution", "provone:Data"],
            "input": ["provone:Execution", "provone:Program", "workflow:Large_Language_Models", "provone:Data"],
            "output": ["provone:Data", "provone:Execution", "workflow:Large_Language_Models", "provone:Program"],
            "program": ["provone:Program", "provone:Execution"],
            "execution": ["provone:Execution", "provone:Program"],
            "agent": ["provone:User", "prov:Agent", "provone:Execution"],
            "time": ["provone:Execution", "workflow:Large_Language_Models"],
            "explanation": ["provone:Data", "provone:Execution", "provone:Program", "workflow:Large_Language_Models"],
        }
        ordered_types: list[str] = []
        for intent in intents:
            ordered_types.extend(type_priority.get(intent, []))
        ordered_types.extend(type_priority["explanation"])

        for preferred_type in ordered_types:
            matching_entities = [
                entity
                for entity in relevant_entities
                if preferred_type in entity.types
            ]
            if matching_entities:
                return max(matching_entities, key=lambda entity: entity.score)

        return relevant_entities[0] if relevant_entities else None

    def _support_entity_for(
        self,
        primary_iri: str,
        relevant_entities: Sequence[RelevantEntity],
        preferred_types: set[str],
    ) -> Optional[str]:
        for entity in relevant_entities:
            candidate_iri = self.expand_curie(entity.id)
            if candidate_iri == primary_iri:
                continue
            if preferred_types.intersection(entity.types):
                return candidate_iri
        return None

    def _describe_entity(self, iri: str, intents: set[str]) -> str:
        record = self.entities.get(iri)
        if record is None:
            return ""

        type_set = set(record.types)
        if "workflow:Large_Language_Models" in type_set:
            return self._describe_llm(iri, intents)
        if "provone:Execution" in type_set:
            return self._describe_execution(iri, intents)
        if "provone:Program" in type_set:
            return self._describe_program(iri, intents)
        if type_set.intersection({"provone:Data", "provone:Collection", "workflow:Large_Language_Model_Output"}):
            return self._describe_data_like_entity(iri, intents)
        if type_set.intersection({"provone:User", "prov:Agent"}):
            return self._describe_agent(iri, intents)

        if "explanation" in intents:
            return self._describe_generic_entity(iri)
        return ""

    def _describe_execution(self, iri: str, intents: Optional[set[str]] = None) -> str:
        intents = intents or set()
        record = self.entities[iri]
        identifier = self._literal_for_predicate(iri, IDENTIFIER_PREDICATES)
        label = f"execution {identifier}" if identifier else record.label
        program_iri = self._program_for_execution(iri)
        program_label = self.entities[program_iri].label if program_iri in self.entities else "an unknown program"
        user_iri = self._first_object(iri, "prov:wasAssociatedWith")
        user_label = self.entities[user_iri].label if user_iri in self.entities else None
        start_time = self._literal_for_predicate(iri, {"prov:startedAt"})
        end_time = self._literal_for_predicate(iri, {"prov:endedAt"})
        input_labels = [self.entities[data_iri].label for data_iri in self._usage_entities_for_execution(iri) if data_iri in self.entities]
        output_labels = [self.entities[data_iri].label for data_iri in self._generated_entities_for_execution(iri) if data_iri in self.entities]
        model_names = self._models_for_execution(iri)
        upstream_programs = [
            self.entities[upstream_program].label
            for upstream_program in self._upstream_programs_for_execution(iri)
            if upstream_program in self.entities
        ]

        sentence = f"{label.capitalize()} ran {program_label}"
        details: list[str] = []
        if model_names:
            details.append(f"using model {', '.join(model_names)}")
        if user_label:
            details.append(f"for {user_label}")
        if start_time and end_time:
            details.append(f"from {start_time} to {end_time}")
        elif start_time:
            details.append(f"starting at {start_time}")
        if input_labels and (
            not intents.intersection({"output", "time"})
            or "input" in intents
            or "execution" in intents
        ):
            details.append(f"with inputs {', '.join(_unique(input_labels[:4]))}")
        if output_labels and ("input" not in intents or intents.intersection({"output", "explanation"})):
            details.append(f"and produced {', '.join(_unique(output_labels[:4]))}")
        if details:
            sentence = f"{sentence} " + ", ".join(details)
        sentence += "."
        return sentence

    def _describe_program(self, iri: str, intents: Optional[set[str]] = None) -> str:
        record = self.entities[iri]
        input_ports = [self.entities[port_iri].label for port_iri in self._objects(iri, "provone:hasInPort") if port_iri in self.entities]
        output_ports = [self.entities[port_iri].label for port_iri in self._objects(iri, "provone:hasOutPort") if port_iri in self.entities]
        execution_ids = []
        for execution_iri in self._executions_for_program(iri):
            identifier = self._literal_for_predicate(execution_iri, IDENTIFIER_PREDICATES)
            execution_ids.append(identifier or self.entities[execution_iri].label)
        description = record.descriptions[0] if record.descriptions else ""
        parts = [f"{record.label} is a workflow program"]
        if description:
            parts.append(f"described as {description}")
        if input_ports:
            parts.append(f"with input ports {', '.join(_unique(input_ports[:4]))}")
        if output_ports:
            parts.append(f"and output ports {', '.join(_unique(output_ports[:4]))}")
        if execution_ids:
            parts.append(f"linked to executions {', '.join(_unique(execution_ids[:4]))}")
        return ", ".join(parts) + "."

    def _describe_llm(self, iri: str, intents: Optional[set[str]] = None) -> str:
        intents = intents or set()
        record = self.entities[iri]
        model_name = self._literal_for_predicate(iri, MODEL_PREDICATES) or record.label
        input_labels = [self.entities[data_iri].label for data_iri in self._objects(iri, "sio:SIO_000230") if data_iri in self.entities]
        output_labels = []
        for llm_output_iri in self._objects(iri, "sio:SIO_000229"):
            output_labels.extend(
                self.entities[target_iri].label
                for target_iri in self._objects(llm_output_iri, "sio:SIO_000202")
                if target_iri in self.entities
            )
        execution_labels = []
        for execution_iri in self._executions_for_llm(iri):
            identifier = self._literal_for_predicate(execution_iri, IDENTIFIER_PREDICATES)
            execution_labels.append(identifier or self.entities[execution_iri].label)
        sentence = f"The model node {record.label} uses {model_name}"
        details: list[str] = []
        if input_labels and (
            not intents.intersection({"output", "time"})
            or "input" in intents
            or "model" in intents
        ):
            details.append(f"and consumes {', '.join(_unique(input_labels[:4]))}")
        if output_labels and ("input" not in intents or intents.intersection({"output", "explanation", "model"})):
            details.append(f"while producing {', '.join(_unique(output_labels[:4]))}")
        if execution_labels:
            details.append(f"during executions {', '.join(_unique(execution_labels[:4]))}")
        if details:
            sentence = f"{sentence} " + ", ".join(details)
        return sentence + "."

    def _describe_data_like_entity(self, iri: str, intents: Optional[set[str]] = None) -> str:
        intents = intents or set()
        record = self.entities[iri]
        preview = self._literal_for_predicate(iri, {"prov:value"})
        producer_execution = self._first_object(iri, "prov:wasGeneratedBy")
        consumer_executions = list(self._subjects("prov:used", iri))
        consumer_programs = [
            self.entities[program_iri].label
            for execution_iri in consumer_executions
            for program_iri in [self._program_for_execution(execution_iri)]
            if program_iri and program_iri in self.entities
        ]

        parts = [f"{record.label.capitalize()} is a data entity"]
        if preview and not intents.intersection({"input", "time"}):
            parts.append(f"with value \"{_truncate(preview, 160)}\"")
        if producer_execution:
            execution_label = self._execution_summary_label(producer_execution)
            program_iri = self._program_for_execution(producer_execution)
            if program_iri and program_iri in self.entities:
                parts.append(
                    f"produced by {execution_label} of {self.entities[program_iri].label}"
                )
            else:
                parts.append(f"produced by {execution_label}")
        if consumer_programs:
            parts.append(f"used by {', '.join(_unique(consumer_programs[:4]))}")
        elif consumer_executions:
            consumer_labels = [self._execution_summary_label(execution_iri) for execution_iri in consumer_executions]
            parts.append(f"used by {', '.join(_unique(consumer_labels[:4]))}")
        return ", ".join(parts) + "."

    def _describe_agent(self, iri: str, intents: Optional[set[str]] = None) -> str:
        record = self.entities[iri]
        execution_labels = [
            self._execution_summary_label(execution_iri)
            for execution_iri in self._subjects("prov:wasAssociatedWith", iri)
        ]
        if execution_labels:
            return f"{record.label.capitalize()} is associated with executions {', '.join(_unique(execution_labels[:5]))}."
        return f"{record.label.capitalize()} is an agent node in the workflow graph."

    def _describe_generic_entity(self, iri: str) -> str:
        record = self.entities[iri]
        facts = [
            f"{predicate} -> {text}"
            for predicate, text in record.literal_facts[:3]
        ]
        if facts:
            return f"{record.label.capitalize()} is connected to the following facts: {'; '.join(facts)}."
        return f"{record.label.capitalize()} is a relevant entity in the workflow graph."

    def _fallback_from_evidence(self, evidence: Sequence[EvidenceTriple]) -> str:
        statements = []
        for triple in evidence:
            object_text = triple.object_label if triple.object_is_literal else triple.object_label
            statements.append(
                f"{triple.subject_label} {triple.predicate_label} {object_text}"
            )
        return "The most relevant graph facts are: " + "; ".join(statements) + "."

    def _program_for_execution(self, execution_iri: str) -> Optional[str]:
        association_iri = self._first_object(execution_iri, "prov:qualifiedAssociation")
        if not association_iri:
            return None
        return self._first_object(association_iri, "prov:hadPlan")

    def _usage_entities_for_execution(self, execution_iri: str) -> list[str]:
        entities: list[str] = []
        for usage_iri in self._objects(execution_iri, "prov:qualifiedUsage"):
            entities.extend(self._objects(usage_iri, "provone:hadEntity"))
        return _unique(entities)

    def _generated_entities_for_execution(self, execution_iri: str) -> list[str]:
        entities: list[str] = []
        for generation_iri in self._objects(execution_iri, "prov:qualifiedGeneration"):
            entities.extend(self._objects(generation_iri, "provone:hadEntity"))
        return _unique(entities)

    def _executions_for_program(self, program_iri: str) -> list[str]:
        executions: list[str] = []
        for association_iri in self._subjects("prov:hadPlan", program_iri):
            executions.extend(self._subjects("prov:qualifiedAssociation", association_iri))
        return _unique(executions)

    def _executions_for_llm(self, llm_iri: str) -> list[str]:
        executions: list[str] = []
        for task_iri in self._subjects("prov:used", llm_iri):
            executions.extend(self._objects(task_iri, "sio:SIO_000313"))
            executions.extend(self._subjects("sio:SIO_000369", task_iri))
        return _unique(executions)

    def _models_for_execution(self, execution_iri: str) -> list[str]:
        models: list[str] = []
        task_iris = self._objects(execution_iri, "sio:SIO_000369")
        task_iris.extend(self._subjects("sio:SIO_000313", execution_iri))
        for task_iri in _unique(task_iris):
            for llm_iri in self._objects(task_iri, "prov:used"):
                model_name = self._literal_for_predicate(llm_iri, MODEL_PREDICATES)
                if model_name:
                    models.append(model_name)
        return _unique(models)

    def _upstream_programs_for_execution(self, execution_iri: str) -> list[str]:
        programs: list[str] = []
        for upstream_execution in self._objects(execution_iri, "prov:wasInformedBy"):
            program_iri = self._program_for_execution(upstream_execution)
            if program_iri:
                programs.append(program_iri)
        return _unique(programs)

    def _execution_summary_label(self, execution_iri: str) -> str:
        identifier = self._literal_for_predicate(execution_iri, IDENTIFIER_PREDICATES)
        if identifier:
            return f"execution {identifier}"
        record = self.entities.get(execution_iri)
        return record.label if record else self.to_curie(execution_iri)

    def _first_object(self, subject_iri: str, predicate_curie: str) -> Optional[str]:
        objects = self._objects(subject_iri, predicate_curie)
        return objects[0] if objects else None

    def _objects(self, subject_iri: str, predicate_curie: str) -> list[str]:
        predicate_iri = self.expand_curie(predicate_curie)
        values = [
            str(obj)
            for obj in self.kg.objects(URIRef(subject_iri), URIRef(predicate_iri))
            if isinstance(obj, URIRef)
        ]
        return _unique(values)

    def _subjects(self, predicate_curie: str, object_iri: str) -> list[str]:
        predicate_iri = self.expand_curie(predicate_curie)
        values = [
            str(subject)
            for subject in self.kg.subjects(URIRef(predicate_iri), URIRef(object_iri))
            if isinstance(subject, URIRef)
        ]
        return _unique(values)

    def _literal_for_predicate(self, subject_iri: str, predicate_ids: Iterable[str]) -> Optional[str]:
        for predicate_id in predicate_ids:
            predicate_iri = self.expand_curie(predicate_id) if ":" in predicate_id else predicate_id
            for literal in self.kg.objects(URIRef(subject_iri), URIRef(predicate_iri)):
                if isinstance(literal, Literal):
                    cleaned = _clean_literal_text(str(literal))
                    if cleaned:
                        return cleaned
        return None

    def _predicate_label(self, predicate_curie: str) -> str:
        schema_info = self.schema_metadata.get(predicate_curie)
        if schema_info:
            return schema_info.get("label", predicate_curie)
        if predicate_curie == "workflow:llm_model":
            return "uses model"
        if predicate_curie in {"prov:startedAt", "prov:endedAt", "prov:atTime"}:
            return _humanize_identifier(predicate_curie.split(":", 1)[1])
        return _humanize_identifier(predicate_curie.split(":", 1)[-1])

    def _object_label(self, obj: URIRef | Literal) -> str:
        if isinstance(obj, Literal):
            return _clean_literal_text(str(obj))
        record = self.entities.get(str(obj))
        if record:
            return record.label
        return self.to_curie(str(obj))

    def _first_literal(
        self,
        graph: Graph,
        subject_iri: str,
        predicate_ids: Iterable[str],
    ) -> Optional[str]:
        for predicate_id in predicate_ids:
            for literal in graph.objects(URIRef(subject_iri), URIRef(predicate_id)):
                if isinstance(literal, Literal):
                    cleaned = _clean_literal_text(str(literal))
                    if cleaned:
                        return cleaned
        return None

    def expand_curie(self, value: str) -> str:
        if value.startswith(("http://", "https://", "urn:")):
            return value
        if ":" not in value:
            return value
        prefix, local = value.split(":", 1)
        namespace = self.namespaces.get(prefix)
        if namespace:
            return f"{namespace}{local}"
        return value

    def to_curie(self, iri: str) -> str:
        best_match: Optional[tuple[str, str]] = None
        for prefix, namespace in self.namespaces.items():
            if iri.startswith(namespace):
                if best_match is None or len(namespace) > len(best_match[1]):
                    best_match = (prefix, namespace)
        if best_match is None:
            return iri
        prefix, namespace = best_match
        return f"{prefix}:{iri[len(namespace):]}"


def _clean_literal_text(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"@[\w-]+\^\^<[^>]+>$", "", cleaned)
    cleaned = re.sub(r"\^\^<[^>]+>$", "", cleaned)
    return cleaned.strip()


def _local_name(value: str) -> str:
    if "#" in value:
        return value.rsplit("#", 1)[1]
    if "/" in value:
        return value.rstrip("/").rsplit("/", 1)[1]
    return value


def _humanize_identifier(value: str) -> str:
    text = value.replace("_", " ").replace("-", " ")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_text(value: str) -> str:
    normalized = _humanize_identifier(_clean_literal_text(value)).lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _normalize_token(token: str) -> str:
    token = token.lower().strip()
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def _tokenize(value: str) -> list[str]:
    tokens = []
    for token in _normalize_text(value).split():
        normalized = _normalize_token(token)
        if normalized and normalized not in STOPWORDS:
            tokens.append(normalized)
    return tokens


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    trimmed = value[:limit].rstrip()
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return f"{trimmed}..."


def _unique[T](items: Iterable[T]) -> list[T]:
    seen: set[Any] = set()
    unique_items: list[T] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items
