from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, TypeVar, Tuple

import dspy
from pydantic import BaseModel

from src.config.experiment import ApplicationInfo, TTLConfig
from src.config.object_search import ObjectSearchConfig
from src.embeddings.base import BaseEmbeddings
from src.explainer.object_search import ObjectSearch
from src.judge.node_processing import AnsweredSignature
from src.llm.base import BaseLLM
from src.synthetic_questions.SQRetriver import SQRetriver
from src.templates.dependancy_graph import (
    AnswerRevisionSignature,
    BuildTopologyGraphSignature,
    SubQuestionSignature,
    SubQuestionVerificationSignature,
    SummarySignature,
    SyntheticQuestionGroundingSignature,
    SyntheticQuestionParameterSignature,
    SyntheticQuestionPlanningSignature,
    SyntheticQuestionResultSignature,
)
from src.utils.adjacency_matrix import build_adjacency_matrix, incoming_edges
from src.utils.graph_manager import GraphManager
from src.utils.utils import clean_string_list, regex_add_strings
from src.vector_db.base import BaseVectorDB

logger = logging.getLogger(__name__)

PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class QuestionNode(BaseModel):
    id: str
    question: str
    node_type: Optional[str] = None

    def solve(
        self,
        schema_info: str,
        predecessor_info: Dict[str, Any],
        runtime: Optional["DependancyGraph"] = None,  
        max_tries: int = 5,
    ) -> Dict[str, Any]:
        if runtime is None:
            return {
                "answer": "",
                "plan": [],
                "execution": [],
                "judge": [],
            }

        predecessor_context = runtime.format_predecessor_context(predecessor_info)
        solution: Dict[str, Any] = {
            "question": self.question,
            "predecessor_context": predecessor_context,
            "draft_answer": "",
            "draft_judge": [],
            "plan": [],
            "grounding": {},
            "candidate_synthetic_questions": [],
            "execution": [],
            "judge": [],
        }

        if predecessor_context:
            with dspy.context(lm=runtime.llm.llm):
                summary_prediction = runtime.summary_predictor(
                    qa_dialog=predecessor_context,
                    original_question=self.question,
                )
            draft_answer = str(getattr(summary_prediction, "answer", "")).strip()
            solution["draft_answer"] = draft_answer
            if draft_answer:
                draft_judge = runtime.ensure_answer_quality(
                    question=self.question,
                    answer=draft_answer,
                    evidence_context=predecessor_context,
                    max_tries=max_tries,
                )
                solution["draft_judge"] = draft_judge["history"]
                if draft_judge["answered"]:
                    solution["answer"] = draft_judge["answer"]
                    return solution
                
        

        plan_bundle = runtime.plan_synthetic_question_execution(
            question=self.question,
            schema_context=schema_info,
            predecessor_context=predecessor_context,
        )
        solution["grounding"] = plan_bundle["grounding"]
        solution["candidate_synthetic_questions"] = plan_bundle["candidate_synthetic_questions"]
        solution["plan"] = plan_bundle["plan"]

        execution_bundle = runtime.execute_synthetic_question_plan(
            question=self.question,
            predecessor_context=predecessor_context,
            plan=plan_bundle["plan"],
        )
        solution["execution"] = execution_bundle["steps"]

        evidence_blocks = [
            predecessor_context.strip(),
            runtime.format_step_results(execution_bundle["steps"]),
        ]
        evidence_context = "\n\n".join(
            block for block in evidence_blocks if block
        ).strip()

        final_answer = execution_bundle["answer"].strip()
        if not final_answer and predecessor_context:
            final_answer = solution["draft_answer"]

        judged_answer = runtime.ensure_answer_quality(
            question=self.question,
            answer=final_answer,
            evidence_context=evidence_context,
            max_tries=max_tries,
        )
        solution["judge"] = judged_answer["history"]
        solution["answer"] = judged_answer["answer"]
        return solution


T = TypeVar("T")
Edge = Tuple[T, T]


class DependancyGraph:
    def __init__(
        self,
        graph_loc: str,
        app_info: ApplicationInfo,
        llm: BaseLLM,
        embedder: BaseEmbeddings,
        vector_db: BaseVectorDB,
        object_search_config: ObjectSearchConfig,
        ttl_config: TTLConfig,
    ) -> None:
        self.vertices: Dict[str, QuestionNode] = {}

        self.app_info = app_info

        self.graph_manager = GraphManager(
            graph_file=graph_loc,
            config=ttl_config,
        )

        self.llm = llm
        self.embedder = embedder
        self.vector_db = vector_db

        self.object_db = ObjectSearch(
            self.graph_manager,
            self.embedder,
            self.vector_db,
            object_search_config,
        )
        self.synthetic_question_retriever = SQRetriver()
        self.synthetic_questions_by_program_id = {
            row["program_id"]: row
            for row in self.synthetic_question_retriever.rows
            if row.get("program_id")
        }

        self.information_required_predictor = dspy.Predict(SubQuestionSignature)
        self.filter_sub_question_predictor = dspy.Predict(
            SubQuestionVerificationSignature
        )
        self.build_topology_graph_predictor = dspy.Predict(
            BuildTopologyGraphSignature
        )
        self.summary_predictor = dspy.Predict(SummarySignature)
        self.synthetic_question_grounding_predictor = dspy.Predict(
            SyntheticQuestionGroundingSignature
        )
        self.synthetic_question_plan_predictor = dspy.Predict(
            SyntheticQuestionPlanningSignature
        )
        self.synthetic_question_parameter_predictor = dspy.Predict(
            SyntheticQuestionParameterSignature
        )
        self.synthetic_question_result_predictor = dspy.Predict(
            SyntheticQuestionResultSignature
        )
        self.answer_revision_predictor = dspy.Predict(AnswerRevisionSignature)
        self.answer_judge_predictor = dspy.Predict(AnsweredSignature)

    def information_required(
        self,
        query: str,
        schema_context: str,
    ) -> List[str]:
        application_context = (self.app_info.description or "").strip()

        with dspy.context(lm=self.llm.llm):
            prediction = self.information_required_predictor(
                user_query=query.strip(),
                schema_context=schema_context.strip(),
                application_context=application_context,
            )

            clean_sub_questions = clean_string_list(
                getattr(prediction, "sub_questions", [])
            )

            if not clean_sub_questions:
                return []

            prediction = self.filter_sub_question_predictor(
                original_question=query,
                sub_questions=[
                    f"{i}). {q}" for i, q in enumerate(clean_sub_questions)
                ],
            )

        filtered_indexes = getattr(prediction, "filtered_sub_question", [])
        try:
            filtered_index_set = {int(index) for index in filtered_indexes}
        except Exception:
            filtered_index_set = set()

        return [
            question
            for index, question in enumerate(clean_sub_questions)
            if index not in filtered_index_set
        ]

    def build_toplevel_dependancy_graph(
        self,
        user_query: str,
        info_req: List[QuestionNode],
    ) -> None:
        self.vertices.update({vertex.id: vertex for vertex in info_req})
        self.vertices["0"] = QuestionNode(id="0", question=user_query)
        self.in_degree = [0] * len(self.vertices)
        self.out_degree = [0] * len(self.vertices)

        with dspy.context(lm=self.llm.llm):
            graph_content = self.build_topology_graph_predictor(
                original_question=user_query.strip(),
                sub_questions=[
                    "{}. {}".format(vertex.id, vertex.question)
                    for vertex in info_req
                ],
            )
            topo_graph_rep = getattr(graph_content, "topology_graph", None)

        if not topo_graph_rep:
            raise ValueError("Failed to build topology graph.")

        self.adjacency_matrix, self.edges = build_adjacency_matrix(topo_graph_rep)

        for source_node, row in enumerate(self.adjacency_matrix):
            for dest_node, has_edge in enumerate(row):
                if not has_edge:
                    continue
                self.out_degree[source_node] += 1
                self.in_degree[dest_node] += 1

        for node_id, vertex in self.vertices.items():
            idx = int(node_id)
            if self.in_degree[idx] == 0:
                vertex.node_type = "leaf"
            elif self.out_degree[idx] == 0:
                vertex.node_type = "root"
            else:
                vertex.node_type = "inner"

    def user_query_to_requirements(
        self,
        query: str,
        schema_context: str = "",
    ) -> Dict[str, Any]:
        info_req = self.information_required(query, schema_context)
        self.build_toplevel_dependancy_graph(
            query,
            [
                QuestionNode(
                    id=str(index + 1),
                    question=sub_question,
                )
                for index, sub_question in enumerate(info_req)
            ],
        )

        return {
            "user_query": query,
            "information_required": info_req,
        }

    def plan_synthetic_question_execution(
        self,
        question: str,
        schema_context: str,
        predecessor_context: str = "",
    ) -> Dict[str, Any]:
        grounding = self.ground_synthetic_questions(
            question=question,
            schema_context=schema_context,
            predecessor_context=predecessor_context,
        )
        candidate_rows = self.collect_candidate_synthetic_questions(
            question=question,
            candidate_classes=grounding["candidate_classes"],
            candidate_relations=grounding["candidate_relations"],
        )

        if not candidate_rows:
            return {
                "grounding": grounding,
                "candidate_synthetic_questions": [],
                "plan": [],
            }

        with dspy.context(lm=self.llm.llm):
            prediction = self.synthetic_question_plan_predictor(
                question=question,
                schema_context=schema_context,
                predecessor_context=predecessor_context,
                candidate_synthetic_questions=self.format_candidate_synthetic_questions(
                    candidate_rows
                ),
            )

        raw_plan = str(getattr(prediction, "execution_plan_json", "")).strip()
        plan = self.parse_plan_json(raw_plan)
        available_program_ids = {
            row["program_id"]
            for row in candidate_rows
            if row.get("program_id")
        }

        filtered_plan = []
        for index, step in enumerate(plan, start=1):
            program_id = str(step.get("program_id", "")).strip()
            if program_id not in available_program_ids:
                continue

            filtered_plan.append(
                {
                    "step_id": str(step.get("step_id", f"step{index}")).strip()
                    or f"step{index}",
                    "sub_question": str(
                        step.get("sub_question", question)
                    ).strip()
                    or question,
                    "program_id": program_id,
                    "input_bindings": self.normalize_bindings(
                        step.get("input_bindings", {})
                    ),
                    "expected_classes": clean_string_list(
                        step.get("expected_classes", [])
                    ),
                }
            )

        if not filtered_plan:
            fallback_row = candidate_rows[0]
            fallback_expected_classes = clean_string_list(
                [
                    fallback_row.get("focal_node") or "",
                    fallback_row.get("start_node") or "",
                    fallback_row.get("end_node") or "",
                ]
            )
            filtered_plan = [
                {
                    "step_id": "step1",
                    "sub_question": question,
                    "program_id": fallback_row["program_id"],
                    "input_bindings": {},
                    "expected_classes": fallback_expected_classes,
                }
            ]

        return {
            "grounding": grounding,
            "candidate_synthetic_questions": candidate_rows,
            "plan": filtered_plan,
        }

    def execute_synthetic_question_plan(
        self,
        question: str,
        predecessor_context: str,
        plan: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        executed_steps: List[Dict[str, Any]] = []

        for step in plan:
            executed_step = self.execute_synthetic_question_step(
                question=question,
                predecessor_context=predecessor_context,
                step=step,
                previous_steps=executed_steps,
            )
            executed_steps.append(executed_step)

        evidence = self.format_step_results(executed_steps)
        evidence_blocks = [
            predecessor_context.strip(),
            evidence,
        ]
        summary_context = "\n\n".join(
            block for block in evidence_blocks if block
        ).strip()

        final_answer = ""
        if summary_context:
            with dspy.context(lm=self.llm.llm):
                prediction = self.summary_predictor(
                    qa_dialog=summary_context,
                    original_question=question,
                )
            final_answer = str(getattr(prediction, "answer", "")).strip()

        return {
            "steps": executed_steps,
            "answer": final_answer,
        }

    def execute_synthetic_question_step(
        self,
        question: str,
        predecessor_context: str,
        step: Dict[str, Any],
        previous_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        row = self.synthetic_questions_by_program_id.get(step["program_id"])
        if row is None:
            return {
                "step_id": step["step_id"],
                "sub_question": step["sub_question"],
                "program_id": step["program_id"],
                "answer": "",
                "important_entities": [],
                "parameter_values": [],
                "results": {},
            }

        parameter_sets = self.resolve_step_parameters(
            question=question,
            predecessor_context=predecessor_context,
            step=step,
            row=row,
            previous_steps=previous_steps,
        )
        query_results: Dict[str, List[Dict[str, Any]]] = {}

        for index, parameter_values in enumerate(parameter_sets, start=1):
            rendered_query = regex_add_strings(row["code"], **parameter_values)
            results_df = self.graph_manager.query(
                rendered_query,
                add_header_tail=self.query_needs_header(rendered_query),
                resolve_curie=True,
            )
            query_results[f"set_{index}"] = results_df.to_dict("records")

        if not query_results:
            query_results["set_1"] = []

        with dspy.context(lm=self.llm.llm):
            prediction = self.synthetic_question_result_predictor(
                question=question,
                predecessor_context=predecessor_context,
                previous_step_results=self.format_step_results(previous_steps),
                step_spec=self.format_step_spec(step, row),
                sparql_results=self.format_query_results(query_results),
            )

        answer = str(getattr(prediction, "answer", "")).strip()
        important_entities = clean_string_list(
            getattr(prediction, "important_entities", [])
        )
        if not important_entities:
            important_entities = self.extract_entities_from_records(query_results)

        if not answer:
            if any(query_results.values()):
                answer = "Retrieved results for the current synthetic question."
            else:
                answer = "No matching results were retrieved for the current synthetic question."

        return {
            "step_id": step["step_id"],
            "sub_question": step["sub_question"],
            "program_id": step["program_id"],
            "parameter_values": parameter_sets,
            "results": query_results,
            "answer": answer,
            "important_entities": important_entities,
        }

    def ground_synthetic_questions(
        self,
        question: str,
        schema_context: str,
        predecessor_context: str = "",
    ) -> Dict[str, Any]:
        available_classes = self.get_available_synthetic_question_classes()
        available_relations = self.get_available_synthetic_question_relations()

        with dspy.context(lm=self.llm.llm):
            prediction = self.synthetic_question_grounding_predictor(
                question=question,
                schema_context=schema_context,
                predecessor_context=predecessor_context,
                available_classes=available_classes,
                available_relations=available_relations,
            )

        raw_candidate_classes = clean_string_list(
            getattr(prediction, "candidate_classes", [])
        )
        raw_candidate_relations = clean_string_list(
            getattr(prediction, "candidate_relations", [])
        )
        entity_phrases = clean_string_list(
            getattr(prediction, "entity_phrases", [])
        )

        available_class_set = set(available_classes)
        available_relation_set = set(available_relations)

        candidate_classes = [
            candidate
            for candidate in raw_candidate_classes
            if candidate in available_class_set
        ]
        if not candidate_classes:
            candidate_classes = self.lexical_match_terms(
                question,
                available_classes,
            )

        candidate_relations = [
            candidate
            for candidate in raw_candidate_relations
            if candidate in available_relation_set
        ]
        if not candidate_relations:
            candidate_relations = self.lexical_match_terms(
                question,
                available_relations,
            )

        if not entity_phrases:
            entity_phrases = self.extract_literal_phrases(question)

        return {
            "candidate_classes": candidate_classes,
            "candidate_relations": candidate_relations,
            "entity_phrases": entity_phrases,
        }

    def collect_candidate_synthetic_questions(
        self,
        question: str,
        candidate_classes: List[str],
        candidate_relations: List[str],
        limit: int = 25,
    ) -> List[Dict[str, Optional[str]]]:
        collected_rows: Dict[str, Dict[str, Optional[str]]] = {}

        def add_rows(rows: List[Dict[str, Optional[str]]]) -> None:
            for row in rows:
                program_id = row.get("program_id")
                if not program_id:
                    continue
                collected_rows[program_id] = row

        for candidate_class in candidate_classes:
            add_rows(
                self.synthetic_question_retriever.retrieve_object_level_from_object(
                    focal_node=candidate_class
                )
            )
            add_rows(
                self.synthetic_question_retriever.retrieve_object_level_from_prop(
                    focal_node=candidate_class
                )
            )

        for candidate_relation in candidate_relations:
            relation_rows = [
                row
                for row in self.synthetic_question_retriever.rows
                if row.get("focal_relation") == candidate_relation
            ]
            if candidate_classes:
                relation_rows = [
                    row
                    for row in relation_rows
                    if row.get("focal_node") in candidate_classes
                ]
            add_rows(relation_rows)

        for start_class in candidate_classes:
            for end_class in candidate_classes:
                add_rows(
                    self.synthetic_question_retriever.retrieve_path_level(
                        start_node=start_class,
                        end_node=end_class,
                    )
                )
            add_rows(
                [
                    row
                    for row in self.synthetic_question_retriever.rows
                    if row.get("category") == "path-level"
                    and (
                        row.get("start_node") == start_class
                        or row.get("end_node") == start_class
                    )
                ]
            )

        candidate_rows = list(collected_rows.values())
        if not candidate_rows:
            candidate_rows = list(self.synthetic_question_retriever.rows)

        ranked_rows = sorted(
            candidate_rows,
            key=lambda row: self.rank_synthetic_question_row(
                row,
                question=question,
                candidate_classes=candidate_classes,
                candidate_relations=candidate_relations,
            ),
            reverse=True,
        )
        return ranked_rows[:limit]

    def resolve_step_parameters(
        self,
        question: str,
        predecessor_context: str,
        step: Dict[str, Any],
        row: Dict[str, Optional[str]],
        previous_steps: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        placeholders = self.extract_query_placeholders(row["code"] or "")
        if not placeholders:
            placeholders = list(self.parse_serialized_dict(row.get("input_spec")).keys())

        direct_candidates = self.build_direct_parameter_candidates(
            step=step,
            placeholders=placeholders,
            previous_steps=previous_steps,
        )
        linked_candidates = self.build_linked_parameter_candidates(
            step=step,
            placeholders=placeholders,
            row=row,
            previous_steps=previous_steps,
            question=question,
        )

        candidate_parameter_values = {
            key: direct_candidates.get(key, []) + linked_candidates.get(key, [])
            for key in placeholders
        }

        with dspy.context(lm=self.llm.llm):
            prediction = self.synthetic_question_parameter_predictor(
                question=question,
                predecessor_context=predecessor_context,
                previous_step_results=self.format_step_results(previous_steps),
                step_spec=self.format_step_spec(step, row),
                candidate_parameter_values=json.dumps(
                    candidate_parameter_values,
                    indent=2,
                ),
            )

        parameter_values = self.parse_parameter_values_json(
            str(getattr(prediction, "parameter_values_json", "")).strip()
        )
        normalized_parameter_sets = self.normalize_parameter_sets(
            parameter_values,
            placeholders=placeholders,
        )

        if normalized_parameter_sets:
            return normalized_parameter_sets

        fallback_parameter_set: Dict[str, str] = {}
        for placeholder in placeholders:
            fallback_values = candidate_parameter_values.get(placeholder, [])
            if not fallback_values:
                continue
            fallback_parameter_set[placeholder] = str(fallback_values[0])

        return [fallback_parameter_set] if fallback_parameter_set else []

    def build_direct_parameter_candidates(
        self,
        step: Dict[str, Any],
        placeholders: List[str],
        previous_steps: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        bindings = step.get("input_bindings", {}) or {}
        candidates = {placeholder: [] for placeholder in placeholders}

        for placeholder in placeholders:
            raw_binding = bindings.get(placeholder)
            if raw_binding is None and len(placeholders) == 1:
                raw_binding = bindings.get("value")
            if raw_binding is None:
                continue

            if isinstance(raw_binding, str) and raw_binding.startswith("STEP:"):
                step_id = raw_binding.split(":", 1)[1].strip()
                previous_step = next(
                    (
                        item
                        for item in previous_steps
                        if item.get("step_id") == step_id
                    ),
                    None,
                )
                if previous_step is not None:
                    candidates[placeholder].extend(
                        previous_step.get("important_entities", [])
                    )
                continue

            candidates[placeholder].append(str(raw_binding))

        return candidates

    def build_linked_parameter_candidates(
        self,
        step: Dict[str, Any],
        placeholders: List[str],
        row: Dict[str, Optional[str]],
        previous_steps: List[Dict[str, Any]],
        question: str,
    ) -> Dict[str, List[str]]:
        candidates = {placeholder: [] for placeholder in placeholders}
        expected_classes = clean_string_list(step.get("expected_classes", []))
        class_hints = expected_classes or clean_string_list(
            [
                row.get("focal_node") or "",
                row.get("start_node") or "",
                row.get("end_node") or "",
            ]
        )
        bindings = step.get("input_bindings", {}) or {}

        for placeholder in placeholders:
            if placeholder == "class_uri":
                for class_hint in class_hints:
                    candidates[placeholder].append(
                        self.graph_manager.resolve_curie(
                            class_hint,
                            allow_bare=True,
                        )
                    )
                continue

            lookup_phrases = []
            direct_value = bindings.get(placeholder)
            if isinstance(direct_value, str) and not direct_value.startswith("STEP:"):
                lookup_phrases.append(direct_value)
            lookup_phrases.append(step.get("sub_question", ""))
            lookup_phrases.append(question)

            if placeholder in {"obj", "obj_uri"}:
                for phrase in clean_string_list(lookup_phrases):
                    linked_entities = self.object_db.link_entities(
                        phrase,
                        class_hints=class_hints,
                    )
                    candidates[placeholder].extend(
                        [
                            str(entity.get("object_uri") or entity.get("object_name"))
                            for entity in linked_entities
                            if entity.get("object_uri") or entity.get("object_name")
                        ]
                    )

                for previous_step in previous_steps:
                    candidates[placeholder].extend(
                        previous_step.get("important_entities", [])
                    )
                continue

            if placeholder == "prop_uri":
                candidates[placeholder].extend(
                    clean_string_list(
                        lookup_phrases + self.extract_literal_phrases(question)
                    )
                )

        return {
            key: clean_string_list(values)
            for key, values in candidates.items()
        }

    def ensure_answer_quality(
        self,
        question: str,
        answer: str,
        evidence_context: str,
        max_tries: int = 2,
    ) -> Dict[str, Any]:
        current_answer = answer.strip()
        history = []

        for _ in range(max(max_tries, 1)):
            with dspy.context(lm=self.llm.llm):
                prediction = self.answer_judge_predictor(
                    question=question,
                    answer=current_answer,
                )

            answered = bool(getattr(prediction, "answered", False))
            feedback = str(getattr(prediction, "feedback", "")).strip()
            history.append(
                {
                    "answered": answered,
                    "feedback": feedback,
                    "answer": current_answer,
                }
            )
            if answered:
                return {
                    "answer": current_answer,
                    "answered": True,
                    "history": history,
                }

            with dspy.context(lm=self.llm.llm):
                revision = self.answer_revision_predictor(
                    question=question,
                    answer=current_answer,
                    feedback=feedback,
                    evidence_context=evidence_context,
                )
            revised_answer = str(
                getattr(revision, "revised_answer", "")
            ).strip()
            if not revised_answer or revised_answer == current_answer:
                break
            current_answer = revised_answer

        return {
            "answer": current_answer,
            "answered": False,
            "history": history,
        }

    @staticmethod
    def solve_node(
        adj_matrix: List[List[int]],
        node_id: int,
        schema_info: str,
        node_map: Dict[str, QuestionNode],
        runtime: "DependancyGraph",
        max_tries: int = 5,
        cache: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if cache is None:
            cache = {}

        node_key = str(node_id)
        if node_key in cache:
            return cache[node_key]

        predecessor_nodes = incoming_edges(adj_matrix, node_id)
        predecessor_info: Dict[str, Dict[str, Any]] = {}
        for predecessor_node in predecessor_nodes:
            predecessor_key = str(predecessor_node)
            predecessor_info[predecessor_key] = DependancyGraph.solve_node(
                adj_matrix,
                predecessor_node,
                schema_info,
                node_map,
                runtime=runtime,
                max_tries=max_tries,
                cache=cache,
            )

        node_data = node_map[node_key].solve(
            schema_info,
            predecessor_info,
            runtime=runtime,
            max_tries=max_tries,
        )
        node_info = {
            "id": node_key,
            "question": node_map[node_key].question,
            "schema_info_used": schema_info,
            "retrieved_objects": [
                entity
                for execution_step in node_data.get("execution", [])
                for entity in execution_step.get("important_entities", [])
            ],
            "synthetic_questions_plan": node_data.get("plan"),
            "intermediary_results": node_data.get("execution"),
            "answer": node_data.get("answer", ""),
            "judge": node_data.get("judge", []),
            "grounding": node_data.get("grounding", {}),
        }
        cache[node_key] = node_info
        return node_info

    def process_dependancy_graph(
        self,
        schema_context: str,
    ) -> Dict[str, Any]:
        if not hasattr(self, "adjacency_matrix"):
            raise ValueError(
                "Dependency graph has not been built. Run `user_query_to_requirements` first."
            )

        solved = DependancyGraph.solve_node(
            self.adjacency_matrix,
            0,
            schema_context,
            self.vertices,
            runtime=self,
        )
        self.last_result = solved
        return solved

    def get_available_synthetic_question_classes(self) -> List[str]:
        classes = {
            row.get(key)
            for row in self.synthetic_question_retriever.rows
            for key in ("focal_node", "start_node", "end_node")
            if row.get(key)
        }
        return sorted(str(value) for value in classes if value)

    def get_available_synthetic_question_relations(self) -> List[str]:
        relations = {
            row.get("focal_relation")
            for row in self.synthetic_question_retriever.rows
            if row.get("focal_relation")
        }
        return sorted(str(value) for value in relations if value)

    def lexical_match_terms(
        self,
        question: str,
        terms: List[str],
        limit: int = 3,
    ) -> List[str]:
        question_terms = {
            token
            for token in re.split(r"[^a-zA-Z0-9]+", question.lower())
            if token
        }
        ranked_terms = []
        for term in terms:
            alias = term.split(":")[-1].replace("_", " ").replace("-", " ")
            alias_terms = {
                token
                for token in re.split(r"[^a-zA-Z0-9]+", alias.lower())
                if token
            }
            overlap = len(question_terms.intersection(alias_terms))
            if overlap:
                ranked_terms.append((overlap, term))

        ranked_terms.sort(key=lambda item: (-item[0], item[1]))
        return [term for _, term in ranked_terms[:limit]]

    def rank_synthetic_question_row(
        self,
        row: Dict[str, Optional[str]],
        question: str,
        candidate_classes: List[str],
        candidate_relations: List[str],
    ) -> int:
        score = 0
        question_terms = {
            token
            for token in re.split(r"[^a-zA-Z0-9]+", question.lower())
            if token
        }

        row_text = " ".join(
            str(row.get(key, ""))
            for key in (
                "program_id",
                "statement",
                "focal_node",
                "start_node",
                "end_node",
                "focal_relation",
            )
        ).lower()

        for candidate_class in candidate_classes:
            if candidate_class == row.get("focal_node"):
                score += 6
            if candidate_class == row.get("start_node") or candidate_class == row.get("end_node"):
                score += 5
            if candidate_class.lower() in row_text:
                score += 2

        for candidate_relation in candidate_relations:
            if candidate_relation == row.get("focal_relation"):
                score += 6
            if candidate_relation.lower() in row_text:
                score += 2

        score += sum(1 for token in question_terms if token in row_text)
        if row.get("category") == "path-level":
            score += 1
        return score

    def format_candidate_synthetic_questions(
        self,
        rows: List[Dict[str, Optional[str]]],
    ) -> str:
        formatted_rows = []
        for row in rows:
            placeholders = self.extract_query_placeholders(row.get("code") or "")
            formatted_rows.append(
                "program_id={program_id} | category={category} | statement={statement} | "
                "start_node={start_node} | end_node={end_node} | focal_node={focal_node} | "
                "focal_relation={focal_relation} | placeholders={placeholders}".format(
                    program_id=row.get("program_id", ""),
                    category=row.get("category", ""),
                    statement=row.get("statement", ""),
                    start_node=row.get("start_node", ""),
                    end_node=row.get("end_node", ""),
                    focal_node=row.get("focal_node", ""),
                    focal_relation=row.get("focal_relation", ""),
                    placeholders=",".join(placeholders),
                )
            )
        return "\n".join(formatted_rows)

    def format_step_spec(
        self,
        step: Dict[str, Any],
        row: Dict[str, Optional[str]],
    ) -> str:
        return json.dumps(
            {
                "step_id": step.get("step_id"),
                "sub_question": step.get("sub_question"),
                "program_id": step.get("program_id"),
                "category": row.get("category"),
                "statement": row.get("statement"),
                "expected_classes": step.get("expected_classes", []),
                "input_bindings": step.get("input_bindings", {}),
                "placeholders": self.extract_query_placeholders(row.get("code") or ""),
            },
            indent=2,
        )

    def format_query_results(
        self,
        query_results: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        return json.dumps(query_results, indent=2)

    def format_step_results(
        self,
        steps: List[Dict[str, Any]],
    ) -> str:
        if not steps:
            return ""

        lines = []
        for step in steps:
            lines.append(f"Step {step.get('step_id', '')}: {step.get('sub_question', '')}")
            lines.append(f"Program: {step.get('program_id', '')}")
            lines.append(f"Answer: {step.get('answer', '')}")
            lines.append(
                "Important entities: "
                + ", ".join(step.get("important_entities", []))
            )
            lines.append(
                "Results: " + json.dumps(step.get("results", {}), indent=2)
            )
        return "\n".join(lines)

    def format_predecessor_context(
        self,
        predecessor_info: Dict[str, Any],
    ) -> str:
        if not predecessor_info:
            return ""

        lines = []
        for predecessor_key, predecessor_value in sorted(predecessor_info.items()):
            lines.append(
                f"Node {predecessor_key}: {predecessor_value.get('question', '')}"
            )
            lines.append(f"Answer: {predecessor_value.get('answer', '')}")
            synthetic_plan = predecessor_value.get("synthetic_questions_plan", [])
            if synthetic_plan:
                lines.append(
                    "Plan steps: "
                    + ", ".join(
                        step.get("program_id", "")
                        for step in synthetic_plan
                        if step.get("program_id")
                    )
                )
        return "\n".join(lines)

    def extract_entities_from_records(
        self,
        query_results: Dict[str, List[Dict[str, Any]]],
    ) -> List[str]:
        collected_values = []
        for records in query_results.values():
            for record in records:
                for value in record.values():
                    text_value = str(value).strip()
                    if not text_value:
                        continue
                    if ":" in text_value or text_value.startswith("http"):
                        collected_values.append(text_value)
        return clean_string_list(collected_values)

    @staticmethod
    def parse_serialized_dict(raw_value: Optional[str]) -> Dict[str, Any]:
        if not raw_value:
            return {}

        try:
            parsed = ast.literal_eval(raw_value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def parse_plan_json(raw_plan: str) -> List[Dict[str, Any]]:
        payload = DependancyGraph.extract_json_payload(raw_plan)
        if payload is None:
            return []

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return []

        if isinstance(parsed, dict):
            parsed = [parsed]
        return [item for item in parsed if isinstance(item, dict)]

    @staticmethod
    def parse_parameter_values_json(raw_json: str) -> List[Dict[str, Any]]:
        payload = DependancyGraph.extract_json_payload(raw_json)
        if payload is None:
            return []

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return []

        if isinstance(parsed, dict):
            parsed = [parsed]
        return [item for item in parsed if isinstance(item, dict)]

    @staticmethod
    def extract_json_payload(raw_text: str) -> Optional[str]:
        stripped_text = raw_text.strip()
        if not stripped_text:
            return None

        if stripped_text.startswith("{") or stripped_text.startswith("["):
            return stripped_text

        match = re.search(r"```json\s*([\s\S]*?)\s*```", stripped_text)
        if match:
            return match.group(1).strip()

        return None

    @staticmethod
    def extract_query_placeholders(query_text: str) -> List[str]:
        return clean_string_list(PLACEHOLDER_PATTERN.findall(query_text))

    @staticmethod
    def normalize_bindings(bindings: Any) -> Dict[str, Any]:
        if not isinstance(bindings, dict):
            return {}
        return {
            str(key).strip(): value
            for key, value in bindings.items()
            if str(key).strip()
        }

    def normalize_parameter_sets(
        self,
        parameter_sets: List[Dict[str, Any]],
        placeholders: List[str],
    ) -> List[Dict[str, str]]:
        normalized_sets = []
        for parameter_set in parameter_sets:
            normalized_set: Dict[str, str] = {}
            for placeholder in placeholders:
                if placeholder not in parameter_set:
                    continue

                value = parameter_set[placeholder]
                if value is None:
                    continue

                text_value = str(value).strip()
                if not text_value:
                    continue

                if placeholder == "class_uri":
                    text_value = self.graph_manager.resolve_curie(
                        text_value,
                        allow_bare=True,
                    )
                elif placeholder in {"obj", "obj_uri"}:
                    text_value = self.graph_manager.resolve_curie(
                        text_value,
                        allow_bare=True,
                    )
                normalized_set[placeholder] = text_value

            if normalized_set:
                normalized_sets.append(normalized_set)

        return normalized_sets

    @staticmethod
    def query_needs_header(query_text: str) -> bool:
        return "PREFIX " not in query_text.upper()

    @staticmethod
    def extract_literal_phrases(question: str) -> List[str]:
        quoted_fragments = re.findall(r'"([^"]+)"', question)
        identifier_like = re.findall(
            r"\b[A-Za-z0-9_:-]{3,}\b",
            question,
        )
        return clean_string_list(quoted_fragments + identifier_like[:5])
