from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

import dspy
from icecream import ic
from pydantic import BaseModel
import logging

from src.config.experiment import ApplicationInfo, TTLConfig
from src.config.object_search import ObjectSearchConfig
from src.embeddings.base import BaseEmbeddings
from src.explainer.object_search import ObjectSearch
from src.judge.node_processing import AnsweredSignature
from src.llm.base import BaseLLM
from src.synthetic_questions.SQRetriver import SQRetriver, SyntheticQuestionCategory
from src.templates.demos.dependancy_graph import (
    build_information_required_fewshot_examples,
)
from src.templates.dependancy_graph import (
    AnswerRevisionSignature,
    BuildTopologyGraphSignature,
    SubQuestionSignature,
    SubQuestionVerificationSignature,
    SummarySignature,
    SyntheticQuestionCategorySelectionSignature,
    SyntheticQuestionGroundingSignature,
    SyntheticQuestionInitialRetrievalDecisionSignature,
    SyntheticQuestionNextStepSignature,
    SyntheticQuestionParameterSignature,
    SyntheticQuestionPlanningSignature,
    SyntheticQuestionResultSignature,
)
from src.templates.node_processing import (
    SchemaAnswerSignature,
    SchemaAnswerabilitySignature,
)
from src.utils.adjacency_matrix import build_adjacency_matrix, incoming_edges
from src.utils.graph_manager import GraphManager
from src.utils.utils import clean_string_list, regex_add_strings
from src.vector_db.base import BaseVectorDB

PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
logger = logging.getLogger(__name__)

class QuestionNode(BaseModel):
    id: str
    question: str
    node_type: Optional[str] = None

    def solve(
        self,
        schema_info: str,
        predecessor_info: Dict[str, Any],
        runtime: DependencyGraphRuntime,
        max_tries: int = 5,
        max_rounds: int = 5,
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.id == "0":
            logger.info("Processing Root Node")
            resolved_application_context = runtime.resolve_application_context(
                application_context
            )
            predecessor_context = runtime.format_predecessor_context(predecessor_info)
            draft_answer = runtime.summarize_answer_from_execution(
                original_question=self.question,
                current_question=self.question,
                schema_context=schema_info,
                predecessor_context=predecessor_context,
                steps=[],
                application_context=resolved_application_context,
            )
            judged_answer = runtime.ensure_answer_quality(
                question=self.question,
                answer=draft_answer,
                evidence_context=predecessor_context,
                predecessor_context=predecessor_context,
                max_tries=max_tries,
                application_context=resolved_application_context,
            )
            return {
                "question": self.question,
                "predecessor_context": predecessor_context,
                "draft_answer": draft_answer,
                "draft_judge": [],
                "schema_reasoning": {
                    "relevant_schema_info": [],
                    "draft_answer": "",
                    "judge": [],
                },
                "plan": [],
                "grounding": {},
                "candidate_synthetic_questions": [],
                "execution": [],
                "judge":  judged_answer["history"],
                "step_context": "",
                "answer": judged_answer["answer"],
            }

        logger.info(
            "Processing Question Node %s, Question: %s",
            self.id,
            self.question,
        )

        resolved_application_context = runtime.resolve_application_context(
            application_context
        )
        predecessor_context = runtime.format_predecessor_context(predecessor_info)
        solution: Dict[str, Any] = {
            "question": self.question,
            "predecessor_context": predecessor_context,
            "draft_answer": "",
            "draft_judge": [],
            "schema_reasoning": {
                "relevant_schema_info": [],
                "draft_answer": "",
                "judge": [],
            },
            "plan": [],
            "grounding": {},
            "candidate_synthetic_questions": [],
            "execution": [],
            "judge": [],
        }

        if not predecessor_context:
            predecessor_context = ""

        schema_answer = runtime.answer_question_from_schema(
            question=self.question,
            schema_context=schema_info,
            max_tries=max_tries,
            predecessor_context=predecessor_context,
            application_context=resolved_application_context,
        )
        solution["schema_reasoning"] = {
            "relevant_schema_info": schema_answer["relevant_schema_info"],
            "draft_answer": schema_answer["draft_answer"],
            "judge": schema_answer["history"],
        }
        if schema_answer["answered"]:
            solution["judge"] = schema_answer["history"]
            solution["answer"] = schema_answer["answer"]
            return solution

        runtime_supports_traversal = all(
            [
                callable(
                    getattr(runtime, "select_synthetic_question_to_execute", None)
                ),
                callable(getattr(runtime, "execute_synthetic_question", None)),
                hasattr(runtime, "synthetic_question_grounding_predictor"),
                hasattr(runtime, "object_db"),
            ]
        )
        if not runtime_supports_traversal:
            return self._solve_with_legacy_plan(
                schema_info=schema_info,
                predecessor_context=predecessor_context,
                runtime=runtime,
                solution=solution,
                max_tries=max_tries,
                application_context=resolved_application_context,
            )

        answered = False
        step = 0   
        step_context = ""
        question = self.question
        while answered is False and max_rounds > step:
            executed_question = question
            question_info = runtime.select_synthetic_question_to_execute(
                question=executed_question,
                original_question=self.question,
                schema_context=schema_info,
                step_count=step,
                step_context=step_context,
                predecessor_context=predecessor_context,
                previous_steps=solution["execution"],
                application_context=resolved_application_context,
            )

            selected_step = dict(question_info.get("selected_step") or {})
            if not selected_step:
                break

            solution["grounding"] = question_info.get(
                "grounding",
                solution["grounding"],
            )
            solution["candidate_synthetic_questions"] = question_info.get(
                "candidate_synthetic_questions",
                solution["candidate_synthetic_questions"],
            )
            solution["plan"].append(selected_step)

            execution_bundle = runtime.execute_synthetic_question(
                question=executed_question,
                original_question=self.question,
                schema_context=schema_info,
                predecessor_context=predecessor_context,
                step_question=question_info,
                previous_steps=solution["execution"],
                application_context=resolved_application_context,
            )

            latest_steps = execution_bundle.get("steps", [])
            solution["execution"].extend(latest_steps)

            evidence_blocks = [
                predecessor_context.strip(),
                runtime.format_schema_reasoning(solution["schema_reasoning"]),
                runtime.format_step_results(solution["execution"]),
                step_context.strip(),
            ]
            evidence_context = "\n\n".join(
                block for block in evidence_blocks if block
            ).strip()
            
            final_answer = str(execution_bundle.get("answer", "")).strip()
            if not final_answer and predecessor_context:
                final_answer = solution["draft_answer"]
            if not final_answer:
                final_answer = str(
                    solution["schema_reasoning"].get("draft_answer", "")
                ).strip()

            judged_answer = runtime.ensure_answer_quality(
                question=self.question,
                answer=final_answer,
                evidence_context=evidence_context,
                predecessor_context=predecessor_context,
                max_tries=max_tries,
                application_context=resolved_application_context,
            )
            solution["judge"] = judged_answer["history"]
            solution["answer"] = judged_answer["answer"]
            if judged_answer["answered"]:
                answered = True

            latest_feedback = ""
            if judged_answer["history"]:
                latest_feedback = str(
                    judged_answer["history"][-1].get("feedback", "")
                ).strip()

            if not answered and callable(
                getattr(runtime, "rewrite_question_for_next_step", None)
            ):
                rewritten_question = runtime.rewrite_question_for_next_step(
                    original_question=self.question,
                    current_question=executed_question,
                    schema_context=schema_info,
                    predecessor_context=predecessor_context,
                    step_context=step_context,
                    latest_step_results=runtime.format_step_results(latest_steps),
                    partial_answer=judged_answer["answer"],
                    judge_feedback=latest_feedback,
                    application_context=resolved_application_context,
                )
                question = (
                    str(rewritten_question.get("next_question", "")).strip()
                    or executed_question
                )

            
            step_context = str(
                runtime.build_step_context(
                    original_question=self.question,
                    current_question=executed_question,
                    selected_step=selected_step,
                    latest_steps=latest_steps,
                    execution_history=solution["execution"],
                    partial_answer=judged_answer["answer"],
                    judge_feedback=latest_feedback,
                    existing_step_context=step_context,
                    next_question=question,
                )
            ).strip()

            step += 1

        solution["step_context"] = step_context
        return solution

    def _solve_with_legacy_plan(
        self,
        schema_info: str,
        predecessor_context: str,
        runtime: DependencyGraphRuntime,
        solution: Dict[str, Any],
        max_tries: int,
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        plan_bundle = runtime.plan_synthetic_question_execution(
            question=self.question,
            schema_context=schema_info,
            predecessor_context=predecessor_context,
            application_context=application_context,
        )
        solution["grounding"] = plan_bundle["grounding"]
        solution["candidate_synthetic_questions"] = plan_bundle[
            "candidate_synthetic_questions"
        ]
        solution["plan"] = plan_bundle["plan"]

        execution_bundle = runtime.execute_synthetic_question_plan(
            question=self.question,
            schema_context=schema_info,
            predecessor_context=predecessor_context,
            plan=plan_bundle["plan"],
            application_context=application_context,
        )
        solution["execution"] = execution_bundle["steps"]

        evidence_blocks = [
            predecessor_context.strip(),
            runtime.format_schema_reasoning(solution["schema_reasoning"]),
            runtime.format_step_results(execution_bundle["steps"]),
        ]
        evidence_context = "\n\n".join(
            block for block in evidence_blocks if block
        ).strip()

        final_answer = str(execution_bundle["answer"]).strip()
        if not final_answer and predecessor_context:
            final_answer = solution["draft_answer"]
        if not final_answer:
            final_answer = str(
                solution["schema_reasoning"].get("draft_answer", "")
            ).strip()

        judged_answer = runtime.ensure_answer_quality(
            question=self.question,
            answer=final_answer,
            evidence_context=evidence_context,
            predecessor_context=predecessor_context,
            max_tries=max_tries,
            application_context=application_context,
        )
        solution["judge"] = judged_answer["history"]
        solution["answer"] = judged_answer["answer"]
        return solution



class DependencyGraphRuntime:
    """Unified dependency-graph runtime.

    The previous mixin-based split made behavior hard to trace. This class keeps
    the same public API while putting the runtime logic back in one place.
    """

    def __init__(
        self,
        graph_loc: str,
        app_info: ApplicationInfo,
        llm: BaseLLM,
        embedder: BaseEmbeddings,
        vector_db: BaseVectorDB,
        object_search_config: ObjectSearchConfig,
        ttl_config: TTLConfig,
        synthetic_questions: str,
    ) -> None:
        self.vertices: Dict[str, QuestionNode] = {}

        self.app_info = app_info
        self.llm = llm
        self.embedder = embedder
        self.vector_db = vector_db

        self.graph_manager = GraphManager(
            graph_file=graph_loc,
            config=ttl_config,
        )
        self.object_db = ObjectSearch(
            self.graph_manager,
            self.embedder,
            self.vector_db,
            object_search_config,
        )
        self.synthetic_question_retriever = SQRetriver(
            synthetic_questions
        )
        self.synthetic_questions_by_program_id = {
            row["program_id"]: row
            for row in self.synthetic_question_retriever.rows
            if row.get("program_id")
        }

        self.information_required_predictor = dspy.Predict(SubQuestionSignature)
        self.information_required_predictor.demos = (
            self._build_information_required_demos()
        )
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
        self.synthetic_question_initial_retrieval_decision_predictor = dspy.Predict(
            SyntheticQuestionInitialRetrievalDecisionSignature
        )
        self.synthetic_question_category_predictor = dspy.Predict(
            SyntheticQuestionCategorySelectionSignature
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
        self.synthetic_question_next_step_predictor = dspy.Predict(
            SyntheticQuestionNextStepSignature
        )
        self.schema_answerability_predictor = dspy.Predict(
            SchemaAnswerabilitySignature
        )
        self.schema_answer_predictor = dspy.Predict(SchemaAnswerSignature)
        self.answer_revision_predictor = dspy.Predict(AnswerRevisionSignature)
        self.answer_judge_predictor = dspy.Predict(AnsweredSignature)

    def resolve_application_context(
        self,
        application_context: Optional[str] = None,
    ) -> str:
        if application_context is not None:
            return str(application_context).strip()

        app_info = getattr(self, "app_info", None)
        if app_info is None:
            return ""

        return str(getattr(app_info, "description", "") or "").strip()

    @staticmethod
    def strip_entity_citations(text: str) -> str:
        return re.sub(
            r"\s*<cite,\s*id=\d+>.*?</cite>",
            "",
            str(text or ""),
            flags=re.DOTALL,
        ).strip()

    def normalize_citation_entities(
        self,
        entities: List[Any],
    ) -> List[str]:
        normalized_entities: List[str] = []
        for entity in entities:
            if isinstance(entity, dict):
                text = str(
                    entity.get("object_name")
                    or entity.get("object_uri")
                    or entity.get("name")
                    or entity.get("id")
                    or ""
                ).strip()
            else:
                text = str(entity or "").strip()

            if text:
                normalized_entities.append(text)

        return clean_string_list(normalized_entities)

    def attach_entity_citations(
        self,
        answer: str,
        entities: List[Any],
        limit: int = 6,
    ) -> str:
        cleaned_answer = self.strip_entity_citations(answer)
        citation_entities = self.normalize_citation_entities(entities)[:limit]
        if not citation_entities:
            return cleaned_answer

        citations = " ".join(
            f"<cite, id={index}>{entity}</cite>"
            for index, entity in enumerate(citation_entities)
        )
        if not cleaned_answer:
            return citations
        return f"{cleaned_answer} {citations}"

    def format_step_output_with_citations(
        self,
        step: Dict[str, Any],
    ) -> Dict[str, Any]:
        formatted_step = dict(step)
        important_entities = self.normalize_citation_entities(
            formatted_step.get("important_entities", [])
        )
        formatted_step["important_entities"] = important_entities
        formatted_step["answer"] = self.attach_entity_citations(
            str(formatted_step.get("answer", "")).strip(),
            important_entities,
        )
        return formatted_step

    def _build_information_required_demos(self) -> List[dspy.Example]:
        demos: List[dspy.Example] = []

        for example in build_information_required_fewshot_examples():
            sub_questions = getattr(example, "sub_questions", None)
            if sub_questions is None:
                sub_questions = getattr(example, "information_required", [])

            demos.append(
                dspy.Example(
                    user_query=getattr(example, "user_query", ""),
                    schema_context=getattr(example, "schema_context", ""),
                    application_context=getattr(
                        example,
                        "application_context",
                        "",
                    ),
                    sub_questions=list(sub_questions),
                ).with_inputs(
                    "user_query",
                    "schema_context",
                    "application_context",
                )
            )

        return demos

    @staticmethod
    def normalize_filtered_sub_question_indices(
        raw_indices: Any,
        question_count: int,
    ) -> List[int]:
        if raw_indices is None:
            return []

        if not isinstance(raw_indices, (list, tuple, set, frozenset)):
            raw_indices = [raw_indices]

        normalized_indices: List[int] = []
        for raw_index in raw_indices:
            if isinstance(raw_index, bool):
                continue

            index_text = str(raw_index).strip()
            if not index_text:
                continue

            try:
                index = int(index_text)
            except ValueError:
                match = re.search(r"-?\d+", index_text)
                if match is None:
                    continue
                index = int(match.group())

            if 0 <= index < question_count and index not in normalized_indices:
                normalized_indices.append(index)

        return normalized_indices

    def format_dependency_node(
        self,
        node_id: int,
        node_map: Optional[Dict[str, QuestionNode]] = None,
    ) -> str:
        resolved_node_map = node_map if node_map is not None else self.vertices
        node_label = "Q" if node_id == 0 else str(node_id)
        node = resolved_node_map.get(str(node_id))
        if node is None:
            return node_label

        question = " ".join(str(node.question).strip().split())
        if not question:
            return node_label
        if len(question) > 80:
            question = question[:77].rstrip() + "..."
        return f"{node_label} [{question}]"

    def format_dependency_path(
        self,
        node_ids: List[int],
        node_map: Optional[Dict[str, QuestionNode]] = None,
    ) -> str:
        return " -> ".join(
            self.format_dependency_node(node_id, node_map=node_map)
            for node_id in node_ids
        )

    @staticmethod
    def outgoing_edges(
        adj_matrix: List[List[int]],
        node_id: int,
    ) -> List[int]:
        return [
            dest_node
            for dest_node, has_edge in enumerate(adj_matrix[node_id])
            if has_edge
        ]

    def find_dependency_cycle(
        self,
        adj_matrix: List[List[int]],
        valid_node_ids: List[int],
    ) -> Optional[List[int]]:
        valid_node_set = set(valid_node_ids)
        visited: set[int] = set()
        active_path: List[int] = []
        active_set: set[int] = set()

        def dfs(node_id: int) -> Optional[List[int]]:
            visited.add(node_id)
            active_path.append(node_id)
            active_set.add(node_id)

            for next_node in self.outgoing_edges(adj_matrix, node_id):
                if next_node not in valid_node_set:
                    continue
                if next_node in active_set:
                    cycle_start_index = active_path.index(next_node)
                    return active_path[cycle_start_index:] + [next_node]
                if next_node in visited:
                    continue
                detected_cycle = dfs(next_node)
                if detected_cycle is not None:
                    return detected_cycle

            active_path.pop()
            active_set.remove(node_id)
            return None

        for node_id in sorted(valid_node_set):
            if node_id in visited:
                continue
            detected_cycle = dfs(node_id)
            if detected_cycle is not None:
                return detected_cycle

        return None

    def validate_dependency_graph(
        self,
        adj_matrix: List[List[int]],
        topology_graph: str,
    ) -> None:
        valid_node_ids = {int(node_id) for node_id in self.vertices}
        invalid_edges: List[tuple[int, int]] = []

        for source_node, row in enumerate(adj_matrix):
            for dest_node, has_edge in enumerate(row):
                if not has_edge:
                    continue
                if source_node not in valid_node_ids or dest_node not in valid_node_ids:
                    invalid_edges.append((source_node, dest_node))

        if invalid_edges:
            invalid_edge_text = ", ".join(
                f"{source}->{dest}"
                for source, dest in invalid_edges[:5]
            )
            raise ValueError(
                "Dependency graph references unknown node ids "
                f"({invalid_edge_text}). Topology graph:\n{topology_graph}"
            )


        detected_cycle = self.find_dependency_cycle(
            adj_matrix,
            list(valid_node_ids),
        )
        if detected_cycle is not None:
            print("detected_cycles")

    def information_required(
        self,
        query: str,
        schema_context: str,
        application_context: Optional[str] = None,
    ) -> List[str]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )

        with dspy.context(lm=self.llm.llm):
            prediction = self.information_required_predictor(
                user_query=query.strip(),
                schema_context=schema_context.strip(),
                application_context=resolved_application_context,
            )

            clean_sub_questions = clean_string_list(
                getattr(prediction, "sub_questions", [])
            )

            if not clean_sub_questions:
                return []

            filter_prediction = self.filter_sub_question_predictor(
                original_question=query.strip(),
                application_context=resolved_application_context,
                sub_questions=[
                    f"{index}). {sub_question}"
                    for index, sub_question in enumerate(clean_sub_questions)
                ],
            )

        filtered_indices = set(
            self.normalize_filtered_sub_question_indices(
                getattr(filter_prediction, "filtered_sub_question", []),
                len(clean_sub_questions),
            )
        )
        if not filtered_indices:
            return clean_sub_questions

        filtered_sub_questions = [
            sub_question
            for index, sub_question in enumerate(clean_sub_questions)
            if index not in filtered_indices
        ]
        logger.info(
            "Filtered sub-questions: kept %s of %s",
            len(filtered_sub_questions),
            len(clean_sub_questions),
        )
        return filtered_sub_questions

    def build_toplevel_dependancy_graph(
        self,
        user_query: str,
        info_req: List[QuestionNode],
        application_context: Optional[str] = None,
    ) -> None:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        self.vertices = {vertex.id: vertex for vertex in info_req}
        self.vertices["0"] = QuestionNode(id="0", question=user_query)

        with dspy.context(lm=self.llm.llm):
            graph_content = self.build_topology_graph_predictor(
                original_question=user_query.strip(),
                application_context=resolved_application_context,
                sub_questions=[
                    "{}. {}".format(vertex.id, vertex.question)
                    for vertex in info_req
                ],
            )
            topo_graph_rep = getattr(graph_content, "topology_graph", None)

        if not topo_graph_rep:
            raise ValueError("Failed to build topology graph.")

        self.topology_graph_rep = str(topo_graph_rep).strip()
        self.adjacency_matrix, self.edges = build_adjacency_matrix(topo_graph_rep)
        self.validate_dependency_graph(
            self.adjacency_matrix,
            self.topology_graph_rep,
        )
        self.in_degree = [0] * len(self.adjacency_matrix)
        self.out_degree = [0] * len(self.adjacency_matrix)

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
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        ic(query)
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        info_req = self.information_required(
            query,
            schema_context,
            application_context=resolved_application_context,
        )
        ic(info_req)
        self.build_toplevel_dependancy_graph(
            query,
            [
                QuestionNode(
                    id=str(index + 1),
                    question=sub_question,
                )
                for index, sub_question in enumerate(info_req)
            ],
            application_context=resolved_application_context,
        )

        return {
            "user_query": query,
            "information_required": info_req,
        }

    @staticmethod
    def solve_node(
        adj_matrix: List[List[int]],
        node_id: int,
        schema_info: str,
        node_map: Dict[str, QuestionNode],
        runtime: "DependencyGraphRuntime",
        max_tries: int = 5,
        cache: Optional[Dict[str, Dict[str, Any]]] = None,
        application_context: Optional[str] = None,
        active_path: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if cache is None:
            cache = {}
        if active_path is None:
            active_path = []

        node_key = str(node_id)

        if node_key in cache:
            return cache[node_key]
        if node_key in active_path:
            cycle_path = active_path[active_path.index(node_key):] + [node_key]
            raise ValueError(
                "Dependency graph cycle encountered during solve: "
                f"{runtime.format_dependency_path([int(value) for value in cycle_path], node_map=node_map)}"
            )

        predecessor_nodes = incoming_edges(adj_matrix, node_id)
        predecessor_info: Dict[str, Dict[str, Any]] = {}
        for predecessor_node in predecessor_nodes:
            predecessor_key = str(predecessor_node)
            predecessor_info[predecessor_key] = type(runtime).solve_node(
                adj_matrix,
                predecessor_node,
                schema_info,
                node_map,
                runtime=runtime,
                max_tries=max_tries,
                cache=cache,
                application_context=application_context,
                active_path=active_path + [node_key],
            )

        node_data = node_map[node_key].solve(
            schema_info,
            predecessor_info,
            runtime=runtime,
            max_tries=max_tries,
            application_context=application_context,
        )
        current_retrieved_objects = clean_string_list(
            [
                entity
                for execution_step in node_data.get("execution", [])
                for entity in execution_step.get("important_entities", [])
            ]
        )
        predecessor_retrieved_objects = clean_string_list(
            [
                entity
                for predecessor_value in predecessor_info.values()
                if isinstance(predecessor_value, dict)
                for entity in predecessor_value.get("retrieved_objects", [])
            ]
        )
        retrieved_objects = clean_string_list(
            current_retrieved_objects + predecessor_retrieved_objects
        )
        answer_entities = (
            current_retrieved_objects
            if current_retrieved_objects
            else predecessor_retrieved_objects
        )
        intermediary_results = [
            runtime.format_step_output_with_citations(step)
            if isinstance(step, dict)
            else step
            for step in node_data.get("execution", [])
        ]
        node_info = {
            "id": node_key,
            "question": node_map[node_key].question,
            "predecessor_info": predecessor_info,
            "predecessor_context": node_data.get("predecessor_context", ""),
            "step_context": node_data.get("step_context", ""),
            "schema_info_used": schema_info,
            "schema_reasoning": node_data.get("schema_reasoning", {}),
            "retrieved_objects": retrieved_objects,
            "synthetic_questions_plan": node_data.get("plan"),
            "intermediary_results": intermediary_results,
            "answer": runtime.attach_entity_citations(
                str(node_data.get("answer", "")).strip(),
                answer_entities,
            ),
            "judge": node_data.get("judge", []),
            "grounding": node_data.get("grounding", {}),
        }
        cache[node_key] = node_info
        return node_info

    def process_dependancy_graph(
        self,
        schema_context: str,
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not hasattr(self, "adjacency_matrix"):
            raise ValueError(
                "Dependency graph has not been built. Run `user_query_to_requirements` first."
            )

        resolved_application_context = self.resolve_application_context(
            application_context
        )
        solved = DependencyGraphRuntime.solve_node(
            self.adjacency_matrix,
            0,
            schema_context,
            self.vertices,
            runtime=self,
            application_context=resolved_application_context,
        )
        self.last_result = solved
        ic(solved)
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
            if (
                candidate_class == row.get("start_node")
                or candidate_class == row.get("end_node")
            ):
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

        flattened_predecessors = self.flatten_predecessor_info(predecessor_info)
        if not flattened_predecessors:
            return ""

        lines = ["Predecessor context:"]
        for predecessor_value in flattened_predecessors:
            predecessor_key = str(
                predecessor_value.get("id", "")
            ).strip()
            question = str(predecessor_value.get("question", "")).strip()
            answer = str(predecessor_value.get("answer", "")).strip()
            schema_context = str(
                predecessor_value.get(
                    "schema_context",
                    predecessor_value.get("schema_info_used", ""),
                )
            ).strip()
            predecessor_step_context = str(
                predecessor_value.get("step_context", "")
            ).strip()
            retrieved_objects = clean_string_list(
                predecessor_value.get("retrieved_objects", [])
            )
            intermediary_results = predecessor_value.get(
                "intermediary_results",
                [],
            )

            lines.append(
                f"Node {predecessor_key}: {question}"
                if predecessor_key
                else f"Question: {question}"
            )
            if answer:
                lines.append(f"Answer: {answer}")

            if schema_context:
                lines.append("Schema context:")
                lines.append(schema_context)

            schema_reasoning = self.format_schema_reasoning(
                predecessor_value.get("schema_reasoning", {})
            )
            if schema_reasoning:
                lines.append(schema_reasoning)

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

            if intermediary_results:
                lines.append("Execution evidence:")
                lines.append(self.format_step_results(intermediary_results))

            if retrieved_objects:
                lines.append(
                    "Retrieved objects: " + ", ".join(retrieved_objects)
                )

            if predecessor_step_context:
                lines.append("Step context:")
                lines.append(predecessor_step_context)

        return "\n".join(lines)

    def flatten_predecessor_info(
        self,
        predecessor_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        flattened: List[Dict[str, Any]] = []
        seen_node_ids = set()

        def sort_key(item: Any) -> Any:
            raw_key = str(item[0])
            return (0, int(raw_key)) if raw_key.isdigit() else (1, raw_key)

        def visit(node_payloads: Dict[str, Any]) -> None:
            for predecessor_key, predecessor_value in sorted(
                node_payloads.items(),
                key=sort_key,
            ):
                if not isinstance(predecessor_value, dict):
                    continue

                nested_predecessors = predecessor_value.get("predecessor_info", {})
                if isinstance(nested_predecessors, dict) and nested_predecessors:
                    visit(nested_predecessors)

                node_id = str(
                    predecessor_value.get("id", predecessor_key)
                ).strip() or str(predecessor_key)
                if node_id in seen_node_ids:
                    continue

                seen_node_ids.add(node_id)
                flattened.append(predecessor_value)

        visit(predecessor_info)
        return flattened

    def format_schema_reasoning(
        self,
        schema_reasoning: Dict[str, Any],
    ) -> str:
        relevant_schema_info = clean_string_list(
            schema_reasoning.get("relevant_schema_info", [])
        )
        if not relevant_schema_info:
            return ""

        lines = ["Schema details:"]
        lines.extend(f"- {schema_fact}" for schema_fact in relevant_schema_info)
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
        payload = DependencyGraphRuntime.extract_json_payload(raw_plan)
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
        payload = DependencyGraphRuntime.extract_json_payload(raw_json)
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

    def select_synthetic_question_execution(
        self,
        question: str,
        schema_context: str,
        step_count: int,
        predecessor_context: str = "",
        step_context: str = "",
        application_context: Optional[str] = None,
        previous_steps: Optional[List[Dict[str, Any]]] = None,
        original_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.select_synthetic_question_to_execute(
            question=question,
            original_question=original_question or question,
            schema_context=schema_context,
            step_count=step_count,
            predecessor_context=predecessor_context,
            step_context=step_context,
            previous_steps=previous_steps,
            application_context=application_context,
        )

    def select_synthetic_question_to_execute(
        self,
        question: str,
        schema_context: str,
        step_count: int,
        predecessor_context: str = "",
        step_context: str = "",
        previous_steps: Optional[List[Dict[str, Any]]] = None,
        original_question: Optional[str] = None,
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        previous_steps = previous_steps or []
        overarching_question = str(original_question or question).strip() or question
        normalized_question = question.strip() or overarching_question
        grounding = self.ground_synthetic_questions(
            question=normalized_question,
            schema_context=schema_context,
            predecessor_context=predecessor_context,
            application_context=resolved_application_context,
        )
        candidate_rows = self.collect_candidate_synthetic_questions(
            question=normalized_question,
            candidate_classes=grounding["candidate_classes"],
            candidate_relations=grounding["candidate_relations"],
            schema_context=schema_context,
            predecessor_context="\n\n".join(
                block for block in [predecessor_context, step_context] if block
            ),
            application_context=resolved_application_context,
        )
        
        linked_entities = self.object_db.link_entities_from_phrases(
            phrases=clean_string_list(
                grounding["entity_phrases"] + [normalized_question]
            ),
            class_hints=grounding["candidate_classes"],
            limit=5,
        )

        first_step_decision = self.decide_initial_retrieval_strategy(
            question=normalized_question,
            schema_context=schema_context,
            candidate_classes=grounding["candidate_classes"],
            entity_phrases=grounding["entity_phrases"],
            linked_entities=linked_entities,
            application_context=resolved_application_context,
        )

        selected_step: Dict[str, Any]
        if (
            step_count == 0
            and first_step_decision["retrieval_mode"] == "linked-entities"
        ):
            selected_step = {
                "step_id": f"step{step_count + 1}",
                "sub_question": normalized_question,
                "program_id": "retrieval::linked-entities",
                "execution_mode": "retrieval",
                "retrieval_mode": "linked-entities",
                "decision_reasoning": first_step_decision["decision_reasoning"],
                "input_bindings": {
                    "lookup_phrases": grounding["entity_phrases"],
                },
                "expected_classes": grounding["candidate_classes"],
                "important_entities": [
                    str(
                        entity.get("object_uri")
                        or entity.get("object_name")
                        or ""
                    )
                    for entity in linked_entities
                ],
            }
        elif (
            step_count == 0
            and first_step_decision["retrieval_mode"] == "class-members"
        ):
            selected_step = {
                "step_id": f"step{step_count + 1}",
                "sub_question": normalized_question,
                "program_id": "explore_object_of_class",
                "execution_mode": "retrieval",
                "retrieval_mode": "class-members",
                "class_member_scope": first_step_decision["class_member_scope"],
                "decision_reasoning": first_step_decision["decision_reasoning"],
                "input_bindings": {
                    "class_uri": grounding["candidate_classes"][0]
                    if grounding["candidate_classes"]
                    else "",
                },
                "expected_classes": grounding["candidate_classes"][:1],
            }
        else:
            candidate_row = self.choose_candidate_synthetic_question_row(
                candidate_rows=candidate_rows,
                previous_steps=previous_steps,
            )
            if candidate_row is None:
                get_generic_object_explorer = getattr(
                    self.synthetic_question_retriever,
                    "get_generic_object_explorer",
                    None,
                )
                generic_object_explorer = (
                    get_generic_object_explorer()
                    if callable(get_generic_object_explorer)
                    else None
                )
                candidate_row = generic_object_explorer

            selected_step = self.build_selected_step_from_row(
                row=candidate_row,
                question=normalized_question,
                step_count=step_count,
                candidate_classes=grounding["candidate_classes"],
                candidate_relations=grounding["candidate_relations"],
                previous_steps=previous_steps,
            )

        return {
            "question": normalized_question,
            "original_question": overarching_question,
            "grounding": grounding,
            "candidate_synthetic_questions": candidate_rows,
            "selected_step": selected_step,
            "linked_entities": linked_entities,
        }

    def execute_synthetic_question(
        self,
        question: str,
        schema_context: str,
        predecessor_context: str,
        step_question: Dict[str, Any],
        previous_steps: Optional[List[Dict[str, Any]]] = None,
        original_question: Optional[str] = None,
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        previous_steps = previous_steps or []
        selected_step = dict(step_question.get("selected_step") or {})
        if not selected_step:
            return {"steps": [], "answer": ""}

        if selected_step.get("execution_mode") == "retrieval":
            executed_step = self.execute_retrieval_step(
                question=question,
                step=selected_step,
                step_question=step_question,
            )
        else:
            executed_step = self.execute_synthetic_question_step(
                question=str(original_question or question).strip() or question,
                predecessor_context=predecessor_context,
                step=selected_step,
                previous_steps=previous_steps,
                application_context=resolved_application_context,
            )

        # summarized_answer = self.summarize_answer_from_execution(
        #     original_question=str(original_question or question).strip() or question,
        #     current_question=question,
        #     schema_context=schema_context,
        #     predecessor_context=predecessor_context,
        #     steps=previous_steps + [executed_step],
        #     application_context=resolved_application_context,
        # )

        return {
            "steps": [executed_step],
            "answer": str(executed_step.get("answer", "")).strip()
        }

    def execute_retrieval_step(
        self,
        question: str,
        step: Dict[str, Any],
        step_question: Dict[str, Any],
    ) -> Dict[str, Any]:
        retrieval_mode = str(step.get("retrieval_mode", "")).strip()
        raw_results: List[Dict[str, Any]] = []
        important_entities: List[str] = []

        if retrieval_mode == "linked-entities":
            raw_results = list(step_question.get("linked_entities") or [])
            important_entities = clean_string_list(
                [
                    entity.get("object_uri") or entity.get("object_name")
                    for entity in raw_results
                ]
            )
        elif retrieval_mode == "class-members":
            class_uri = str(
                step.get("input_bindings", {}).get("class_uri", "")
            ).strip()
            class_member_scope = str(
                step.get("class_member_scope", "")
            ).strip()
            if class_member_scope == "linked-only":
                raw_results = self.filter_linked_entities_for_class(
                    linked_entities=list(
                        step_question.get("linked_entities") or []
                    ),
                    class_uri=class_uri,
                    limit=25,
                )
            if not raw_results:
                raw_results = self.object_db.get_objects_of_class(
                    class_uri,
                    limit=25,
                )
            important_entities = clean_string_list(
                [
                    entity.get("object_uri") or entity.get("object_name")
                    for entity in raw_results
                ]
            )

        answer = (
            f"Retrieved {len(important_entities)} candidate object(s) relevant to "
            f"the question: {question}"
        )
        if not important_entities:
            answer = (
                "No candidate objects were retrieved for the current traversal step."
            )

        return {
            "step_id": step.get("step_id", ""),
            "sub_question": step.get("sub_question", question),
            "program_id": step.get("program_id", ""),
            "execution_mode": "retrieval",
            "parameter_values": [step.get("input_bindings", {})],
            "results": {"set_1": raw_results},
            "answer": answer,
            "important_entities": important_entities,
        }

    def summarize_answer_from_execution(
        self,
        original_question: str,
        current_question: str,
        schema_context: str,
        predecessor_context: str,
        steps: List[Dict[str, Any]],
        schema_reasoning: Optional[Dict[str, Any]] = None,
        step_context: str = "",
        application_context: Optional[str] = None,
    ) -> str:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        evidence_blocks = [
            predecessor_context.strip(),
            self.format_schema_reasoning(schema_reasoning or {}),
            self.format_step_results(steps),
            step_context.strip(),
        ]
        summary_context = "\n\n".join(
            block for block in evidence_blocks if block
        ).strip()
        if not summary_context:
            return ""

        with dspy.context(lm=self.llm.llm):
            prediction = self.summary_predictor(
                qa_dialog=summary_context,
                schema_context=schema_context,
                application_context=resolved_application_context,
                original_question=original_question,
            )

        return str(getattr(prediction, "answer", "")).strip()

    def rewrite_question_for_next_step(
        self,
        original_question: str,
        current_question: str,
        schema_context: str,
        predecessor_context: str,
        step_context: str,
        latest_step_results: str,
        partial_answer: str,
        judge_feedback: str,
        application_context: Optional[str] = None,
    ) -> Dict[str, str]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        next_question = ""

        with dspy.context(lm=self.llm.llm):
            prediction = self.synthetic_question_next_step_predictor(
                original_question=original_question,
                current_question=current_question,
                application_context=resolved_application_context,
                schema_context=schema_context,
                predecessor_context=predecessor_context,
                step_context=step_context,
                latest_step_results=latest_step_results,
                partial_answer=partial_answer,
                judge_feedback=judge_feedback,
            )
        next_question = str(getattr(prediction, "next_question", "")).strip()

        if not next_question:
            latest_focus = judge_feedback.strip() or latest_step_results.strip()
            if latest_focus:
                next_question = (
                    f"Given the current evidence, what should be retrieved next to answer "
                    f"'{original_question}'? Focus on: {latest_focus}"
                )
            else:
                next_question = original_question

        return {"next_question": next_question}

    def build_step_context(
        self,
        original_question: str,
        current_question: str,
        selected_step: Dict[str, Any],
        latest_steps: List[Dict[str, Any]],
        execution_history: List[Dict[str, Any]],
        partial_answer: str,
        judge_feedback: str,
        existing_step_context: str = "",
        next_question: str = "",
    ) -> str:
        latest_step_summary = self.format_step_results(latest_steps)
        lines = []
        if existing_step_context.strip():
            lines.append(existing_step_context.strip())
            lines.append("")
        lines.append(f"Original question: {original_question}")
        lines.append(f"Current traversal question: {current_question}")
        lines.append(f"Selected program: {selected_step.get('program_id', '')}")
        lines.append(
            f"Execution mode: {selected_step.get('execution_mode', 'synthetic')}"
        )
        if latest_step_summary:
            lines.append(latest_step_summary)
        if partial_answer:
            lines.append(f"Current answer: {partial_answer}")
        if judge_feedback:
            lines.append(f"Judge feedback: {judge_feedback}")
        if next_question:
            lines.append(f"Next question: {next_question}")
        if execution_history and not latest_step_summary:
            lines.append(self.format_step_results(execution_history))
        return "\n".join(lines).strip()

    def plan_synthetic_question_execution(
        self,
        question: str,
        schema_context: str,
        predecessor_context: str = "",
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        grounding = self.ground_synthetic_questions(
            question=question,
            schema_context=schema_context,
            predecessor_context=predecessor_context,
            application_context=resolved_application_context,
        )
        candidate_rows = self.collect_candidate_synthetic_questions(
            question=question,
            candidate_classes=grounding["candidate_classes"],
            candidate_relations=grounding["candidate_relations"],
            schema_context=schema_context,
            predecessor_context=predecessor_context,
            application_context=resolved_application_context,
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
                application_context=resolved_application_context,
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
        schema_context: str,
        predecessor_context: str,
        plan: List[Dict[str, Any]],
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        executed_steps: List[Dict[str, Any]] = []

        for step in plan:
            executed_step = self.execute_synthetic_question_step(
                question=question,
                predecessor_context=predecessor_context,
                step=step,
                previous_steps=executed_steps,
                application_context=resolved_application_context,
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
                    schema_context=schema_context,
                    application_context=resolved_application_context,
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
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
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
            application_context=resolved_application_context,
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
                application_context=resolved_application_context,
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

    def answer_question_from_schema(
        self,
        question: str,
        schema_context: str,
        max_tries: int = 2,
        predecessor_context: Optional[str] = None,
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        normalized_schema_context = schema_context.strip()
        empty_response = {
            "answer": "",
            "answered": False,
            "draft_answer": "",
            "history": [],
            "relevant_schema_info": [],
        }

        if not normalized_schema_context:
            return empty_response

        with dspy.context(lm=self.llm.llm):
            prediction = self.schema_answerability_predictor(
                question=question,
                application_context=resolved_application_context,
                schema_context=normalized_schema_context,
                predecessor_context=predecessor_context,
            )

        answerable_from_schema = bool(
            getattr(prediction, "answerable_from_schema", False)
        )
        relevant_schema_info = clean_string_list(
            getattr(prediction, "relevant_schema_info", [])
        )

        if not answerable_from_schema:
            return {
                **empty_response,
                "relevant_schema_info": relevant_schema_info,
            }

        schema_evidence_context = "\n".join(
            f"- {schema_fact}" for schema_fact in relevant_schema_info
        ).strip() or normalized_schema_context

        with dspy.context(lm=self.llm.llm):
            answer_prediction = self.schema_answer_predictor(
                question=question,
                application_context=resolved_application_context,
                predecessor_context=predecessor_context,
                relevant_schema_info=schema_evidence_context,
            )

        draft_answer = str(getattr(answer_prediction, "answer", "")).strip()
        if not draft_answer:
            return {
                **empty_response,
                "relevant_schema_info": relevant_schema_info,
            }

        judged_answer = self.ensure_answer_quality(
            question=question,
            answer=draft_answer,
            evidence_context=schema_evidence_context,
            predecessor_context=predecessor_context,
            max_tries=max_tries,
            application_context=resolved_application_context,
        )

        return {
            "answer": judged_answer["answer"],
            "answered": judged_answer["answered"],
            "draft_answer": draft_answer,
            "history": judged_answer["history"],
            "relevant_schema_info": relevant_schema_info,
        }

    def ground_synthetic_questions(
        self,
        question: str,
        schema_context: str,
        predecessor_context: str = "",
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        available_classes = self.get_available_synthetic_question_classes()
        available_relations = self.get_available_synthetic_question_relations()

        with dspy.context(lm=self.llm.llm):
            prediction = self.synthetic_question_grounding_predictor(
                question=question,
                application_context=resolved_application_context,
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

    def format_linked_entity_candidates(
        self,
        linked_entities: List[Dict[str, Any]],
        limit: int = 5,
    ) -> str:
        lines: List[str] = []
        for entity in linked_entities[:limit]:
            object_name = str(
                entity.get("object_name") or entity.get("object_uri") or ""
            ).strip()
            object_classes = clean_string_list(entity.get("object_class", []))
            source = str(entity.get("source", "")).strip()

            details = [object_name] if object_name else []
            if object_classes:
                details.append(f"classes={', '.join(object_classes[:3])}")
            if source:
                details.append(f"source={source}")
            if details:
                lines.append(f"- {'; '.join(details)}")

        return "\n".join(lines)

    def decide_initial_retrieval_strategy(
        self,
        question: str,
        schema_context: str,
        candidate_classes: List[str],
        entity_phrases: List[str],
        linked_entities: List[Dict[str, Any]],
        application_context: Optional[str] = None,
    ) -> Dict[str, str]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        has_candidate_classes = bool(candidate_classes)
        has_linked_entities = bool(linked_entities)

        if not has_candidate_classes and not has_linked_entities:
            return {
                "retrieval_mode": "linked-entities",
                "class_member_scope": "linked-only",
                "decision_reasoning": "No candidate classes or linked entities were available, so direct linking is the safest fallback.",
            }

        if not has_candidate_classes:
            return {
                "retrieval_mode": "linked-entities",
                "class_member_scope": "linked-only",
                "decision_reasoning": "No candidate class was grounded from the schema, so the step should start from linked entities.",
            }

        if not has_linked_entities:
            return {
                "retrieval_mode": "class-members",
                "class_member_scope": "all",
                "decision_reasoning": "No concrete entity candidates were linked, so the step should explore the grounded class directly.",
            }

        retrieval_mode = ""
        class_member_scope = ""
        decision_reasoning = ""

        with dspy.context(lm=self.llm.llm):
            prediction = self.synthetic_question_initial_retrieval_decision_predictor(
                question=question,
                application_context=resolved_application_context,
                schema_context=schema_context,
                candidate_classes=candidate_classes[:3],
                entity_phrases=entity_phrases[:5],
                linked_entity_candidates=self.format_linked_entity_candidates(
                    linked_entities
                ),
            )

        retrieval_mode = str(
            getattr(prediction, "retrieval_mode", "")
        ).strip()
        class_member_scope = str(
            getattr(prediction, "class_member_scope", "")
        ).strip()
        decision_reasoning = str(
            getattr(prediction, "decision_reasoning", "")
        ).strip()

        if retrieval_mode not in {"linked-entities", "class-members"}:
            retrieval_mode = ""
        if class_member_scope not in {"linked-only", "all"}:
            class_member_scope = ""

        if not retrieval_mode:
            if len(entity_phrases) == 1 and len(linked_entities) <= 3:
                retrieval_mode = "linked-entities"
                class_member_scope = "linked-only"
            else:
                retrieval_mode = "class-members"
                class_member_scope = "linked-only"

        if retrieval_mode == "linked-entities":
            class_member_scope = "linked-only"
        elif not class_member_scope:
            class_member_scope = "linked-only" if has_linked_entities else "all"

        if not decision_reasoning:
            if retrieval_mode == "linked-entities":
                decision_reasoning = (
                    "The question appears to target a concrete linked entity, so direct entity retrieval is the highest-precision first step."
                )
            else:
                decision_reasoning = (
                    "The question appears to require scanning class members before narrowing to the relevant objects."
                )

        return {
            "retrieval_mode": retrieval_mode,
            "class_member_scope": class_member_scope,
            "decision_reasoning": decision_reasoning,
        }

    def filter_linked_entities_for_class(
        self,
        linked_entities: List[Dict[str, Any]],
        class_uri: str,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        if not linked_entities:
            return []

        object_db = getattr(self, "object_db", None)
        matches_class_hints = getattr(
            object_db,
            "_result_matches_class_hints",
            None,
        )
        if callable(matches_class_hints) and class_uri.strip():
            filtered_entities = [
                entity
                for entity in linked_entities
                if matches_class_hints(entity, [class_uri])
            ]
            if filtered_entities:
                return filtered_entities[:limit]

        return linked_entities[:limit]

    def collect_candidate_synthetic_questions(
        self,
        question: str,
        candidate_classes: List[str],
        candidate_relations: List[str],
        schema_context: str = "",
        predecessor_context: str = "",
        application_context: Optional[str] = None,
        limit: int = 25,
    ) -> List[Dict[str, Optional[str]]]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        collected_rows: Dict[str, Dict[str, Optional[str]]] = {}
        selected_categories = self.select_plausible_synthetic_question_categories(
            question=question,
            candidate_classes=candidate_classes,
            candidate_relations=candidate_relations,
            schema_context=schema_context,
            predecessor_context=predecessor_context,
            application_context=resolved_application_context,
        )
        selected_category_set = set(selected_categories)

        def add_rows(rows: List[Dict[str, Optional[str]]]) -> None:
            for row in rows:
                program_id = row.get("program_id")
                if not program_id:
                    continue
                collected_rows[program_id] = row

        if (
            SyntheticQuestionCategory.OBJECT_LEVEL_FROM_OBJECT.value
            in selected_category_set
        ):
            for candidate_class in candidate_classes:
                add_rows(
                    self.synthetic_question_retriever.retrieve_object_level_from_object(
                        focal_node=candidate_class
                    )
                )

        if (
            SyntheticQuestionCategory.OBJECT_LEVEL_FROM_PROP.value
            in selected_category_set
        ):
            for candidate_class in candidate_classes:
                add_rows(
                    self.synthetic_question_retriever.retrieve_object_level_from_prop(
                        focal_node=candidate_class
                    )
                )

        object_level_categories = {
            SyntheticQuestionCategory.OBJECT_LEVEL_FROM_OBJECT.value,
            SyntheticQuestionCategory.OBJECT_LEVEL_FROM_PROP.value,
        }.intersection(selected_category_set)
        for candidate_relation in candidate_relations:
            relation_rows = []
            for category in object_level_categories:
                if category == SyntheticQuestionCategory.OBJECT_LEVEL_FROM_OBJECT.value:
                    relation_rows.extend(
                        self.synthetic_question_retriever.retrieve_object_level_from_object(
                            focal_relation=candidate_relation,
                        )
                    )
                elif category == SyntheticQuestionCategory.OBJECT_LEVEL_FROM_PROP.value:
                    relation_rows.extend(
                        self.synthetic_question_retriever.retrieve_object_level_from_prop(
                            focal_relation=candidate_relation,
                        )
                    )
            if candidate_classes:
                relation_rows = [
                    row
                    for row in relation_rows
                    if row.get("focal_node") in candidate_classes
                ]
            add_rows(relation_rows)

        if SyntheticQuestionCategory.PATH_LEVEL.value in selected_category_set:
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
                        if row.get("category")
                        == SyntheticQuestionCategory.PATH_LEVEL.value
                        and (
                            row.get("start_node") == start_class
                            or row.get("end_node") == start_class
                        )
                    ]
                )

        candidate_rows = list(collected_rows.values())
        if not candidate_rows and selected_category_set:
            candidate_rows = [
                row
                for row in self.synthetic_question_retriever.rows
                if row.get("category") in selected_category_set
            ]

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

    def select_candidate_synthetic_question(
        self,
        question: str,
        candidate_classes: List[str],
        candidate_relations: List[str],
        schema_context: str = "",
        predecessor_context: str = "",
        application_context: Optional[str] = None,
        limit: int = 25,
    ) -> List[Dict[str, Optional[str]]]:
        return self.collect_candidate_synthetic_questions(
            question=question,
            candidate_classes=candidate_classes,
            candidate_relations=candidate_relations,
            schema_context=schema_context,
            predecessor_context=predecessor_context,
            application_context=application_context,
            limit=limit,
        )

    def choose_candidate_synthetic_question_row(
        self,
        candidate_rows: List[Dict[str, Optional[str]]],
        previous_steps: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Optional[str]]]:
        if not candidate_rows:
            return None

        executed_program_ids = {
            str(step.get("program_id", "")).strip()
            for step in previous_steps
            if step.get("program_id")
        }
        unseen_rows = [
            row
            for row in candidate_rows
            if str(row.get("program_id", "")).strip() not in executed_program_ids
        ]
        ranked_pool = unseen_rows or candidate_rows

        if previous_steps:
            for row in ranked_pool:
                placeholders = self.extract_query_placeholders(row.get("code") or "")
                if any(
                    placeholder in {"obj", "obj_uri"}
                    for placeholder in placeholders
                ):
                    return row

        return ranked_pool[0]

    def build_selected_step_from_row(
        self,
        row: Optional[Dict[str, Optional[str]]],
        question: str,
        step_count: int,
        candidate_classes: List[str],
        candidate_relations: List[str],
        previous_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if row is None:
            return {}

        placeholders = self.extract_query_placeholders(row.get("code") or "")
        if not placeholders:
            placeholders = list(
                self.parse_serialized_dict(row.get("input_spec")).keys()
            )

        input_bindings: Dict[str, Any] = {}
        latest_step_id = ""
        if previous_steps:
            latest_step_id = str(
                previous_steps[-1].get("step_id", "")
            ).strip()

        for placeholder in placeholders:
            if placeholder == "class_uri" and candidate_classes:
                input_bindings[placeholder] = candidate_classes[0]
            elif placeholder in {"obj", "obj_uri"}:
                if latest_step_id:
                    input_bindings[placeholder] = f"STEP:{latest_step_id}"
                elif candidate_classes:
                    input_bindings[placeholder] = candidate_classes[0]
            elif placeholder == "prop_uri":
                if row.get("focal_relation"):
                    input_bindings[placeholder] = row.get("focal_relation")
                elif candidate_relations:
                    input_bindings[placeholder] = candidate_relations[0]

        return {
            "step_id": f"step{step_count + 1}",
            "sub_question": question,
            "program_id": str(row.get("program_id", "")).strip(),
            "execution_mode": "synthetic",
            "input_bindings": input_bindings,
            "expected_classes": clean_string_list(
                [
                    row.get("focal_node") or "",
                    row.get("start_node") or "",
                    row.get("end_node") or "",
                ]
            )
            or candidate_classes[:2],
        }

    def get_available_synthetic_question_categories(self) -> List[str]:
        retriever = getattr(self, "synthetic_question_retriever", None)
        if retriever is None:
            return []

        if hasattr(retriever, "get_available_categories"):
            categories = retriever.get_available_categories()
            return [str(category) for category in categories if category]

        categories = {
            row.get("category")
            for row in getattr(retriever, "rows", [])
            if row.get("category")
        }
        known_categories = [
            category
            for category in SyntheticQuestionCategory.values()
            if category in categories
        ]
        if known_categories:
            return known_categories
        return sorted(str(category) for category in categories if category)

    def format_synthetic_question_category_options(
        self,
        categories: List[str],
    ) -> str:
        lines = []
        for category in categories:
            description = category
            retriever = getattr(self, "synthetic_question_retriever", None)
            if retriever is not None and hasattr(retriever, "get_category_description"):
                description = retriever.get_category_description(category)
            else:
                try:
                    description = SyntheticQuestionCategory(category).description
                except ValueError:
                    description = category
            lines.append(f"- {category}: {description}")
        return "\n".join(lines)

    def normalize_synthetic_question_categories(
        self,
        raw_categories: Any,
        available_categories: List[str],
    ) -> List[str]:
        if raw_categories is None:
            return []

        if isinstance(raw_categories, str):
            normalized_text = raw_categories.strip()
            if not normalized_text:
                raw_categories = []
            else:
                try:
                    parsed_categories = json.loads(normalized_text)
                except json.JSONDecodeError:
                    parsed_categories = re.split(r"[\n,]+", normalized_text)
                raw_categories = parsed_categories

        if not isinstance(raw_categories, (list, tuple, set, frozenset)):
            raw_categories = [raw_categories]

        available_category_set = set(available_categories)
        normalized_categories: List[str] = []
        retriever = getattr(self, "synthetic_question_retriever", None)
        for raw_category in raw_categories:
            category_text = str(raw_category).strip()
            if not category_text:
                continue

            normalized_category = category_text
            if retriever is not None and hasattr(retriever, "normalize_category"):
                try:
                    normalized_category = retriever.normalize_category(category_text)
                except ValueError:
                    continue

            if (
                normalized_category in available_category_set
                and normalized_category not in normalized_categories
            ):
                normalized_categories.append(normalized_category)

        return normalized_categories

    def infer_synthetic_question_categories(
        self,
        candidate_classes: List[str],
        candidate_relations: List[str],
        available_categories: List[str],
    ) -> List[str]:
        available_category_set = set(available_categories)
        inferred_categories: List[str] = []

        if candidate_classes:
            for category in (
                SyntheticQuestionCategory.OBJECT_LEVEL_FROM_OBJECT.value,
                SyntheticQuestionCategory.OBJECT_LEVEL_FROM_PROP.value,
                SyntheticQuestionCategory.PATH_LEVEL.value,
            ):
                if (
                    category in available_category_set
                    and category not in inferred_categories
                ):
                    inferred_categories.append(category)

        if candidate_relations:
            for category in (
                SyntheticQuestionCategory.OBJECT_LEVEL_FROM_PROP.value,
                SyntheticQuestionCategory.OBJECT_LEVEL_FROM_OBJECT.value,
            ):
                if (
                    category in available_category_set
                    and category not in inferred_categories
                ):
                    inferred_categories.append(category)

        if inferred_categories:
            return inferred_categories
        return list(available_categories)

    def select_plausible_synthetic_question_categories(
        self,
        question: str,
        candidate_classes: List[str],
        candidate_relations: List[str],
        schema_context: str = "",
        predecessor_context: str = "",
        application_context: Optional[str] = None,
    ) -> List[str]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        available_categories = self.get_available_synthetic_question_categories()
        if len(available_categories) <= 1:
            return available_categories

        prediction = None
        predictor = getattr(self, "synthetic_question_category_predictor", None)
        if predictor is not None and getattr(self, "llm", None) is not None:
            with dspy.context(lm=self.llm.llm):
                prediction = predictor(
                    question=question,
                    application_context=resolved_application_context,
                    schema_context=schema_context,
                    predecessor_context=predecessor_context,
                    candidate_classes=candidate_classes,
                    candidate_relations=candidate_relations,
                    category_options=self.format_synthetic_question_category_options(
                        available_categories
                    ),
                )

        selected_categories = self.normalize_synthetic_question_categories(
            getattr(prediction, "plausible_categories", None),
            available_categories,
        )
        if selected_categories:
            return selected_categories

        return self.infer_synthetic_question_categories(
            candidate_classes=candidate_classes,
            candidate_relations=candidate_relations,
            available_categories=available_categories,
        )

    def resolve_step_parameters(
        self,
        question: str,
        predecessor_context: str,
        step: Dict[str, Any],
        row: Dict[str, Optional[str]],
        previous_steps: List[Dict[str, Any]],
        application_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
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
                application_context=resolved_application_context,
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
        predecessor_context: str = "",
        max_tries: int = 2,
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        current_answer = answer.strip()
        history = []

        for _ in range(max(max_tries, 1)):
            with dspy.context(lm=self.llm.llm):
                prediction = self.answer_judge_predictor(
                    question=question,
                    application_context=resolved_application_context,
                    predecessor_context=predecessor_context,
                    evidence_context=evidence_context,
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
                    application_context=resolved_application_context,
                    predecessor_context=predecessor_context,
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
