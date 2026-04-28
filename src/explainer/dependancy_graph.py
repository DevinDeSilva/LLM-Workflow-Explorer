from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

import dspy
from dspy.utils.usage_tracker import UsageTracker, track_usage
from icecream import ic
from pydantic import BaseModel
import pandas as pd
import logging

from src.config.experiment import ApplicationInfo, TTLConfig
from src.config.object_search import ObjectSearchConfig
from src.embeddings.base import BaseEmbeddings
from src.explainer.object_search import ObjectSearch
from src.judge.node_processing import AnsweredSignature
from src.llm.base import BaseLLM
from src.synthetic_questions.SQRetriver import SQRetriver
from src.templates.demos.dependancy_graph import (
    build_information_required_fewshot_examples,
)
from src.templates.dependancy_graph import (
    BuildTopologyGraphSignature,
    ImportantEntitySelectionSignature,
    SummarySignature,
    InitialDataSyntheticQuestion,
    SelectSyntheticQuestionSignature,
    SyntheticQuestionGroundingSignature,
    SyntheticQuestionNextStepSignature,
    SyntheticQuestionParameterSignature,
    SyntheticQuestionPathGroundingSignature,
    SyntheticQuestionResultSignature,
    UserQueryCoverageRewriteSignature,
)
from src.templates.node_processing import (
    SchemaAnswerSignature,
    SchemaAnswerabilitySignature,
)

from src.utils.graph_manager import GraphManager
from src.utils.utils import clean_string_list, regex_add_strings
from src.vector_db.base import BaseVectorDB

PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
logger = logging.getLogger(__name__)

class QuestionNode(BaseModel):
    id: str
    question: str
    original_question:str
    node_type: Optional[str] = None

    def solve(
        self,
        schema_info: str,
        runtime: DependencyGraphRuntime,
        max_rounds: int = 3,
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:

        logger.info(
            "Processing Question Node %s, Question: %s",
            self.id,
            self.question,
        )

        resolved_application_context = runtime.resolve_application_context(
            application_context
        )
        solution: Dict[str, Any] = {
            "question": self.question,
            "step_details":{},
            "draft_answer": "",
            "draft_judge": [],
            "plan": [],
            "grounding": {},
            "candidate_synthetic_questions": [],
            "execution": [],
            "judge": [],
            "answer": "",
            "context": "",
        }


        initial_data = runtime.select_initial_data(
            question=self.question,
            original_question=self.original_question,
            schema_context=schema_info,
            application_context=resolved_application_context,
        )

        answered = False
        question = self.question
        solution["step_details"]["initial-data"] = initial_data
        solution["grounding"] = initial_data.get("grounding", {})
        
        for step in range(max(max_rounds, 1)):
            executed_question = question

            if step == 0:
                step_data = initial_data
                

            else:
                step_data = runtime.run_synthetic_question_path(
                    step_num=step,
                    question=question,
                    original_question= self.question,
                    schema_context= schema_info,
                    application_context=resolved_application_context,
                    step_context = step_context,
                    judged_answer = solution["judge"],
                    step_data=solution["execution"]
                )


            step_objects = runtime.extract_initial_data_objects(
                step_data
            )
            _step = runtime.build_initial_data_step(
                question=executed_question,
                initial_data=step_data,
                object_uris=step_objects,
                step_id=f"step{len(solution['execution']) + 1}",
            )
            _step["extracted_results"] = runtime.retrieve_attr_data_for_objects(
                object_uris=step_objects
            )
            latest_steps = [_step]
            solution["execution"].extend(latest_steps)
            solution["plan"].append(
                {
                    "round": step,
                    "question": executed_question,
                    "object_uris": step_objects,
                }
            )

            round_context = runtime.format_answer_loop_round_context(
                round_number=step + 1,
                question=executed_question,
                initial_data=step_data,
                object_uris=step_objects,
                latest_steps=latest_steps,
            )

            if step == 0:
                step_context = round_context
            else:
                step_context = "\n\n".join(
                    block for block in [step_context, round_context] if block
                ).strip()

            draft_answer = runtime.summarize_answer_from_execution(
                original_question=self.question,
                current_question=executed_question,
                schema_context=schema_info,
              #  steps=solution["execution"],
                step_context=step_context,
                application_context=resolved_application_context,
            )
            solution["draft_answer"] = draft_answer

            evidence_blocks = [step_context]
            final_evidence_context = "\n\n".join(
                block for block in evidence_blocks if block
            ).strip()

            judged_answer = runtime.ensure_answer_quality(
                question=self.question,
                answer=draft_answer,
                evidence_context=final_evidence_context,
                application_context=resolved_application_context,
            )
            solution["judge"] = judged_answer
            solution["answer"] = judged_answer["answer"]
            if judged_answer["answered"]:
                # should_break = True
                # if step == 0 and 'extracted_results' in _step:
                #     for v in _step['extracted_results']:
                #         for v1 in v['attributes']:
                #             if  v1["relation"] == "rdf:type" and v1['object'] == "provone:Data":
                #                 should_break = False
                                
                # if should_break:
                break
            
            step += 1

            latest_feedback = str(
                judged_answer.get("feedback", "")
            ).strip()

            if not answered:
                rewritten_question = runtime.rewrite_question_for_next_step(
                    original_question=self.question,
                    current_question=executed_question,
                    schema_context=schema_info,
                    step_context=final_evidence_context,
                    # latest_step_results=runtime.format_step_results(latest_steps),
                    partial_answer=judged_answer["answer"],
                    judge_feedback=latest_feedback,
                    application_context=resolved_application_context,
                )
                question = (
                    str(rewritten_question.get("next_question", "")).strip()
                    or executed_question
                )

                # selected_entities = runtime.select_important_entities_for_next_step(
                #     original_question=self.question,
                #     current_question=executed_question,
                #     application_context=resolved_application_context,
                #     step_context=step_context,
                #     judge_context=latest_feedback,
                #     latest_steps=latest_steps,
                #     candidate_entities=step_objects,
                # )
                # if selected_entities["important_entities"]:
                #     latest_steps[0]["important_entities"] = selected_entities[
                #         "important_entities"
                #     ]
                #     step_context = "\n\n".join(
                #         block
                #         for block in [
                #             step_context,
                #             runtime.format_selected_entities_context(
                #                 selected_entities
                #             ),
                #         ]
                #         if block
                #     ).strip()

        solution["step_context"] = step_context
        solution["context"] = final_evidence_context
        solution["answered"] = answered
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
            str(row.get("program_id", "")).strip(): row
            for row in self.synthetic_question_retriever.rows.where(
                self.synthetic_question_retriever.rows.notna(),
                None,
            ).to_dict("records")
            if str(row.get("program_id", "")).strip()
        }

        self.user_query_coverage_rewrite_predictor = dspy.Predict(
            UserQueryCoverageRewriteSignature
        )

        self.build_topology_graph_predictor = dspy.Predict(
            BuildTopologyGraphSignature
        )
        self.summary_predictor = dspy.Predict(SummarySignature)
        self.synthetic_question_grounding_predictor = dspy.Predict(
            SyntheticQuestionGroundingSignature
        )
        self.initial_data_synthetic_question_predictor = dspy.Predict(
            InitialDataSyntheticQuestion
        )
        self.select_synthetic_question_predictor = dspy.Predict(
            SelectSyntheticQuestionSignature
        )
        self.synthetic_question_path_grounding_predictor = dspy.Predict(
            SyntheticQuestionPathGroundingSignature
        )
        self.synthetic_question_parameter_predictor = dspy.Predict(
            SyntheticQuestionParameterSignature
        )
        self.synthetic_question_result_predictor = dspy.Predict(
            SyntheticQuestionResultSignature
        )
        self.important_entity_selection_predictor = dspy.Predict(
            ImportantEntitySelectionSignature
        )
        self.synthetic_question_next_step_predictor = dspy.Predict(
            SyntheticQuestionNextStepSignature
        )
        self.schema_answerability_predictor = dspy.Predict(
            SchemaAnswerabilitySignature
        )
        self.schema_answer_predictor = dspy.Predict(SchemaAnswerSignature)
        self.answer_judge_predictor = dspy.Predict(AnsweredSignature)
        self.last_token_usage: Dict[str, Any] = self.empty_token_usage()

    @staticmethod
    def empty_token_usage() -> Dict[str, Any]:
        return {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "models": {},
            "calls": [],
            "estimated": False,
            "source": "none",
        }

    @staticmethod
    def token_count_from_usage(usage: Dict[str, Any], *keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    @classmethod
    def has_token_counts(cls, usage: Dict[str, Any]) -> bool:
        return bool(
            cls.token_count_from_usage(
                usage,
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
                "input_tokens",
                "output_tokens",
            )
        )

    @classmethod
    def stringify_for_token_count(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            prioritized_parts = [
                cls.stringify_for_token_count(value[key])
                for key in ("role", "content", "text", "reasoning_content")
                if key in value
            ]
            if prioritized_parts:
                return "\n".join(part for part in prioritized_parts if part)
            return json.dumps(value, default=str, ensure_ascii=False)
        if isinstance(value, (list, tuple)):
            return "\n".join(cls.stringify_for_token_count(item) for item in value)
        return str(value)

    @classmethod
    def count_text_tokens(cls, value: Any, model: str = "") -> int:
        text = cls.stringify_for_token_count(value)
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

    @classmethod
    def estimate_token_usage_from_history(
        cls,
        lm: Any,
        history_start: int = 0,
    ) -> List[Dict[str, Any]]:
        history = getattr(lm, "history", []) or []
        history_entries = history[max(history_start, 0):]
        calls: List[Dict[str, Any]] = []

        for entry in history_entries:
            if not isinstance(entry, dict):
                continue

            model = str(
                entry.get("model")
                or getattr(lm, "model", "")
                or "unknown"
            )
            prompt_tokens = cls.count_text_tokens(
                entry.get("messages") or entry.get("prompt") or "",
                model=model,
            )
            completion_tokens = cls.count_text_tokens(
                entry.get("outputs") or "",
                model=model,
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            calls.append(
                {
                    "model": model,
                    "usage": usage,
                    "estimated": True,
                    "source": "dspy_lm_history",
                }
            )

        return calls

    @classmethod
    def aggregate_token_calls(
        cls,
        calls: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, int]]:
        usage_by_model: Dict[str, Dict[str, int]] = {}
        for call in calls:
            model = str(call.get("model") or "unknown")
            usage = call.get("usage", {})
            if not isinstance(usage, dict):
                continue
            model_usage = usage_by_model.setdefault(
                model,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            model_usage["prompt_tokens"] += cls.token_count_from_usage(
                usage,
                "prompt_tokens",
                "input_tokens",
            )
            model_usage["completion_tokens"] += cls.token_count_from_usage(
                usage,
                "completion_tokens",
                "output_tokens",
            )
            total_tokens = cls.token_count_from_usage(usage, "total_tokens")
            if not total_tokens:
                total_tokens = (
                    cls.token_count_from_usage(usage, "prompt_tokens", "input_tokens")
                    + cls.token_count_from_usage(
                        usage,
                        "completion_tokens",
                        "output_tokens",
                    )
                )
            model_usage["total_tokens"] += total_tokens
        return usage_by_model

    @classmethod
    def summarize_token_usage(
        cls,
        tracker: UsageTracker,
        lm: Any = None,
        history_start: int = 0,
    ) -> Dict[str, Any]:
        usage_by_model = tracker.get_total_tokens()
        calls: List[Dict[str, Any]] = []

        for model, usage_entries in tracker.usage_data.items():
            for usage_entry in usage_entries:
                calls.append(
                    {
                        "model": model,
                        "usage": dict(usage_entry),
                        "estimated": False,
                        "source": "provider",
                    }
                )

        if not any(cls.has_token_counts(call["usage"]) for call in calls):
            estimated_calls = cls.estimate_token_usage_from_history(
                lm,
                history_start=history_start,
            )
            if estimated_calls:
                calls = estimated_calls
                usage_by_model = cls.aggregate_token_calls(calls)

        prompt_tokens = sum(
            cls.token_count_from_usage(usage, "prompt_tokens", "input_tokens")
            for usage in usage_by_model.values()
        )
        completion_tokens = sum(
            cls.token_count_from_usage(
                usage,
                "completion_tokens",
                "output_tokens",
            )
            for usage in usage_by_model.values()
        )
        total_tokens = sum(
            cls.token_count_from_usage(usage, "total_tokens")
            for usage in usage_by_model.values()
        )
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens

        estimated = any(call.get("estimated", False) for call in calls)

        return {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "models": dict(usage_by_model),
            "calls": calls,
            "estimated": estimated,
            "source": "dspy_lm_history" if estimated else ("provider" if calls else "none"),
        }

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

    @staticmethod
    def solve_node(
        adj_matrix: List[List[int]],
        node_id: int,
        schema_info: str,
        node_map: Dict[str, QuestionNode],
        runtime: "DependencyGraphRuntime",
        max_tries: int = 3,
        cache: Optional[Dict[str, Dict[str, Any]]] = None,
        application_context: Optional[str] = None,
        active_path: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if cache is None:
            cache = {}
        if active_path is None:
            active_path = []

        node_key = str(node_id)
        predecessor_info: Dict[str, Dict[str, Any]] = {}


        node_data = node_map[node_key].solve(
            schema_info,
            runtime=runtime,
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
        user_query: str,
        schema_context: str,
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        lm_history_start = len(getattr(self.llm.llm, "history", []) or [])
        with track_usage() as token_tracker:
            solved = self._process_dependancy_graph(
                user_query=user_query,
                schema_context=schema_context,
                application_context=application_context,
            )

        token_usage = self.summarize_token_usage(
            token_tracker,
            lm=self.llm.llm,
            history_start=lm_history_start,
        )
        solved["token_usage"] = token_usage
        self.last_token_usage = token_usage
        self.last_result = solved
        ic(solved)
        return solved

    def _process_dependancy_graph(
        self,
        user_query:str,
        schema_context: str,
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )

        self.adjacency_matrix = [[0] * 3 for _ in range(3)]
        # original_user_query = user_query
        # with dspy.context(lm=self.llm.llm):
        #     rewrite_prediction = self.user_query_coverage_rewrite_predictor(
        #         user_query=user_query,
        #         application_context=resolved_application_context,
        #         schema_context=schema_context,
        #     )
        # answer_requirements = clean_string_list(
        #     getattr(rewrite_prediction, "answer_requirements", []) or []
        # )
        # rewritten_user_query = str(
        #     getattr(rewrite_prediction, "rewritten_user_query", "")
        # ).strip()
        # if answer_requirements and "cover" not in rewritten_user_query.lower():
        #     rewritten_user_query = (
        #         f"{rewritten_user_query or user_query}\n"
        #         "To be properly answered, cover: "
        #         + "; ".join(answer_requirements)
        #     )
        # user_query2 = rewritten_user_query or user_query

        self.vertices["0"] = QuestionNode(
            id="0",
                question=user_query,
                original_question=user_query
            )

        solved = DependencyGraphRuntime.solve_node(
            self.adjacency_matrix,
            0,
            schema_context,
            self.vertices,
            runtime=self,
            application_context=resolved_application_context,
        )
        solved["original_question"] = user_query
        # solved["answer_requirements"] = answer_requirements
        return solved

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
            solves = self.synthetic_question_retriever.get_program_by_id(
                step.get('program_id', ''),
                solves_only=True
            )

            #imp_entities = step.get("important_entities", [])
            
            if "extracted_results" in step and len(step["extracted_results"]) > 0:
                
                entity_attr = {x['uri']:x['attributes'] for x in step["extracted_results"]}
                entity_str = ""
                for k,v in entity_attr.items():
                    _i = ""
                    for x in v:
                        #if "-" == x['object_class'] or "rdf:type" == x['relation']:
                        _i += "{}={}".format(x['relation'], x['object'])
                        # if "-" != x['object_class']:
                        #     _i += "[{}]".format(x['object_class'])

                        # if "-" != x['object_label']:
                        #     _i += "({})".format(x['object_label'])

                        _i += '; '

                    entity_str += k + " =>" +_i + "\n"
                    
            else:
                entity_str = json.dumps(step.get("results", {}), indent=2)




            lines.append(f"Step {step.get('step_id', '')}: {step.get('sub_question', '')}")
            lines.append(f"Function: {solves}")
            lines.append(f"Answer: {step.get('answer', '')}")
            lines.append(
               "Important entities: \n"
               + entity_str
            )
            # lines.append(
            #    "Results: " + json.dumps(step.get("results", {}), indent=2)
            # )
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
                if not isinstance(record, dict):
                    continue
                for value in record.values():
                    text_value = str(value).strip()
                    if not text_value:
                        continue
                    if ":" in text_value or text_value.startswith("http"):
                        collected_values.append(text_value)
        return clean_string_list(collected_values)

    @staticmethod
    def looks_like_object_identifier(value: Any) -> bool:
        text_value = str(value or "").strip()
        if not text_value:
            return False
        if text_value.startswith(("http://", "https://", "urn:")):
            return True
        if ":" not in text_value:
            return False
        return not bool(re.search(r"\s", text_value))

    def extract_initial_data_objects(
        self,
        initial_data: Dict[str, Any],
        limit: int = 100,
    ) -> List[str]:
        object_uris: List[str] = []

        def add_candidate(value: Any) -> None:
            if self.looks_like_object_identifier(value):
                object_uris.append(str(value).strip())

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key in (
                    "object_uri",
                    "uri",
                    "value",
                    "obj",
                    "obj_uri",
                    "subject",
                    "s",
                ):
                    if key in value:
                        add_candidate(value.get(key))
                for nested_value in value.values():
                    if isinstance(nested_value, (dict, list, tuple)):
                        visit(nested_value)
                return

            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
                return

            add_candidate(value)

        for source_key in ("linked_entities", "results"):
            if source_key in initial_data:
                visit(initial_data.get(source_key))

        return clean_string_list(object_uris)[:limit]

    def build_initial_data_step(
        self,
        question: str,
        initial_data: Dict[str, Any],
        object_uris: List[str],
        step_id: str,
    ) -> Dict[str, Any]:
        results = initial_data.get("results")
        if results is None:
            results = {"set_1": initial_data.get("linked_entities", [])}

        program_id = str(initial_data.get("program_id", "")).strip()
        if not program_id:
            program_id = (
                "retrieval::linked-entities"
                if initial_data.get("strategy") == "by_linked_data"
                else "initial-data"
            )

        answer = (
            f"Retrieved {len(object_uris)} initial candidate object(s) "
            f"relevant to the question: {question}"
        )
        if not object_uris:
            answer = "No initial candidate objects were retrieved."

        return {
            "step_id": step_id,
            "sub_question": question,
            "program_id": program_id,
            "execution_mode": "initial-data",
            "strategy": initial_data.get("strategy", ""),
            "parameter_values": initial_data.get("parameter_values", []),
            "results": results,
            "answer": answer,
            "important_entities": object_uris,
        }

    @staticmethod
    def normalize_property_output_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize SPARQL output with direct properties and collection-member properties.

        Expected input columns:
        p, o, o_label, o_class, c_p, c_o, c_o_label, c_o_class

        Output columns:
        relation, object, object_label, object_class, source
        """

        if "p" in df.columns:
            direct_df = df[["p", "o", "o_label", "o_class"]].copy()
            direct_df = direct_df.rename(
                columns={
                    "p": "relation",
                    "o": "object",
                    "o_label": "object_label",
                    "o_class": "object_class",
                }
            )
        else:

            direct_df = df[["c_p", "c_o", "c_o_label", "c_o_class"]].copy()
            direct_df = direct_df.rename(
                columns={
                    "c_p": "relation",
                    "c_o": "object",
                    "c_o_label": "object_label",
                    "c_o_class": "object_class",
                }
            )

        return direct_df.drop_duplicates().reset_index(drop=True)

    def retrieve_attr_data_for_objects(
        self,
        object_uris: List[str],
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        attr_row_getter = getattr(
            self.synthetic_question_retriever,
            "get_attr_of_object",
            None,
        )
        if not callable(attr_row_getter):
            return []

        attr_row = attr_row_getter()
        if not attr_row:
            return []

        code = str(attr_row.get("code") or "")
        placeholders = self.extract_query_placeholders(code)
        steps: List[Dict[str, Any]] = []

        for offset, object_uri in enumerate(object_uris[:limit]):
            resolved_object_uri = self.graph_manager.resolve_curie(
                object_uri,
                allow_bare=True,
            )
            parameter_values = {
                placeholder: resolved_object_uri
                for placeholder in placeholders
                if placeholder in {"obj", "obj_uri"}
            }
            if not parameter_values and not placeholders:
                parameter_values = {"obj_uri": resolved_object_uri}

            missing_placeholders = [
                placeholder
                for placeholder in placeholders
                if placeholder not in parameter_values
            ]
            query_results: Dict[str, List[Dict[str, Any]]] = {}
            if missing_placeholders:
                query_results["set_1"] = [
                    {
                        "object_uri": object_uri,
                        "missing_placeholders": ", ".join(missing_placeholders),
                    }
                ]
            else:
                rendered_query = regex_add_strings(code, **parameter_values)
                results_df = self.graph_manager.query(
                    rendered_query,
                    add_header_tail=self.query_needs_header(rendered_query),
                    resolve_curie=True,
                )

                results_df = DependencyGraphRuntime.normalize_property_output_df(
                    results_df
                )
                query_results["set_1"] = results_df.to_dict("records")


            steps.append(

                {"uri":object_uri, "attributes":query_results["set_1"]}
            )

        return steps

    def format_answer_loop_round_context(
        self,
        round_number: int,
        question: str,
        initial_data: Dict[str, Any],
        object_uris: List[str],
        latest_steps: List[Dict[str, Any]],
    ) -> str:
        context_payload = {
            "round": round_number,
            "question": question,
            #"strategy": initial_data.get("strategy", ""),
            "program_id": initial_data.get("program_id", ""),
            #"grounding": initial_data.get("grounding", {}),
            "object_uris": object_uris,
        }
        step_results = self.format_step_results(latest_steps)
        return "\n".join(
            [
                "Answer loop round:",
                json.dumps(context_payload, indent=2, default=str),
                step_results,
            ]
        ).strip()

    def select_important_entities_for_next_step(
        self,
        original_question: str,
        current_question: str,
        application_context: str,
        step_context: str,
        judge_context: str,
        latest_steps: List[Dict[str, Any]],
        candidate_entities: List[str],
    ) -> Dict[str, Any]:
        candidates = clean_string_list(candidate_entities)
        if not candidates:
            return {"important_entities": [], "selection_reasoning": ""}

        with dspy.context(lm=self.llm.llm):
            prediction = self.important_entity_selection_predictor(
                original_question=original_question,
                current_question=current_question,
                application_context=application_context,
                step_context=step_context,
                judge_context=judge_context,
                latest_step_results=self.format_step_results(latest_steps),
                candidate_entities=candidates,
            )

        selected_entities = clean_string_list(
            getattr(prediction, "important_entities", []) or []
        )
        candidate_set = set(candidates)
        selected_entities = [
            entity for entity in selected_entities if entity in candidate_set
        ]
        if not selected_entities:
            selected_entities = candidates[:5]

        return {
            "important_entities": selected_entities,
            "selection_reasoning": str(
                getattr(prediction, "selection_reasoning", "")
            ).strip(),
        }

    @staticmethod
    def format_selected_entities_context(selected_entities: Dict[str, Any]) -> str:
        important_entities = clean_string_list(
            selected_entities.get("important_entities", [])
        )
        if not important_entities:
            return ""

        lines = [
            "Important entities selected for the next step:",
            ", ".join(important_entities),
        ]
        selection_reasoning = str(
            selected_entities.get("selection_reasoning", "")
        ).strip()
        if selection_reasoning:
            lines.append(f"Selection reason: {selection_reasoning}")
        return "\n".join(lines)

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
            collect = []
            for k,v in parsed.items():
                if isinstance(v, list):
                    for i in v:
                        collect.append({k:i})
                
                else:
                    collect.append({k:v})
            parsed = collect
            
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
                if "\n" in text_value:
                    text_value = text_value.split('\n', maxsplit=1)[0]

                if text_value == 'nan':
                    continue

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
        quoted_fragments = DependencyGraphRuntime.extract_quoted_phrases(question)
        identifier_like = re.findall(
            r"\b[A-Za-z0-9_:-]{3,}\b",
            question,
        )
        return clean_string_list(quoted_fragments + identifier_like[:5])

    @staticmethod
    def clean_explicit_uri_candidate(candidate: str) -> str:
        return str(candidate or "").strip().strip("<>").rstrip(".,;!?)]}\"'")

    def extract_explicit_uri_candidates(self, question: str) -> List[str]:
        full_uri_candidates = re.findall(
            r"\b(?:https?://|urn:)[^\s<>\"]+",
            question,
        )
        angle_uri_candidates = re.findall(r"<([^<>\s]+)>", question)
        curie_candidates = re.findall(
            r"(?<![\w/])([A-Za-z][A-Za-z0-9_.-]*:[^\s<>\"]+)",
            question,
        )
        candidates = []
        for candidate in angle_uri_candidates + full_uri_candidates + curie_candidates:
            cleaned_candidate = self.clean_explicit_uri_candidate(candidate)
            if not cleaned_candidate:
                continue
            if cleaned_candidate.startswith(("http:", "https:", "urn:")):
                candidates.append(cleaned_candidate)
                continue
            if ":" not in cleaned_candidate:
                continue
            prefix, local_part = cleaned_candidate.split(":", 1)
            namespaces = getattr(self.graph_manager, "config", {}).get(
                "namespaces",
                {},
            )
            if prefix in namespaces and local_part:
                candidates.append(cleaned_candidate)
        return clean_string_list(candidates)

    def resolve_explicit_uri_candidate(self, candidate: str) -> str:
        if candidate.startswith(("http:", "https:", "urn:")):
            return candidate
        return self.graph_manager.resolve_curie(candidate, allow_bare=False)

    def uri_exists_in_graph(self, uri: str) -> bool:
        query = """
        SELECT ?role WHERE {
            { <{uri}> ?p ?o . BIND("subject" AS ?role) }
            UNION
            { ?s ?p <{uri}> . BIND("object" AS ?role) }
        }
        LIMIT 1
        """
        results_df = self.graph_manager.query(
            regex_add_strings(query, uri=uri),
            add_header_tail=False,
        )
        return not results_df.empty

    def initial_data_from_explicit_uri(
        self,
        question: str,
        original_question: str,
    ) -> Optional[Dict[str, Any]]:
        for candidate in self.extract_explicit_uri_candidates(original_question):
            resolved_uri = self.resolve_explicit_uri_candidate(candidate)
            if not resolved_uri or not self.uri_exists_in_graph(resolved_uri):
                continue

            object_classes = self.object_db.get_object_classes(resolved_uri)
            object_description = self.object_db.build_object_description(resolved_uri)
            object_name = (
                self.graph_manager.reverse_curie(resolved_uri)
                if callable(getattr(self.graph_manager, "reverse_curie", None))
                else candidate
            )
            linked_entity = {
                "object_uri": resolved_uri,
                "object_name": object_name,
                "object_class": object_classes,
                "object_description": object_description,
                "source": "explicit-uri",
            }
            return {
                "question": question,
                "original_question": original_question,
                "grounding": {
                    "candidate_classes": object_classes,
                    "candidate_relations": [],
                    "entity_phrases": [candidate],
                },
                "strategy": "by_explicit_uri",
                "linked_entities": [linked_entity],
            }
        return None


    def run_synthetic_question_path(
            self,
            step_num:int,
            question,
            schema_context: str,
            original_question: str,
            application_context: str,
            step_context:str,
            judged_answer:Dict[str,str],
            step_data:List[Dict[str,Any]]
        ):
        resolved_application_context = self.resolve_application_context(
            application_context
        )

        judge_context = "\n".join([f"{k} = {v}" for k,v in judged_answer.items() if k in ['answer','feedback']])

        overarching_question = str(original_question or question).strip() or question
        normalized_question = question.strip() or overarching_question

        grounding = self.ground_synthetic_questions_path(
            question=normalized_question,
            schema_context=schema_context,
            application_context=resolved_application_context,
            step_context = step_context,
            judge_context = judge_context,
        )

        program_data = self.synthetic_question_path_run(
            question=normalized_question,
            step_num=step_num,
            schema_context=schema_context,
            candidate_classes=grounding["candidate_classes"],
            entity_phrases=grounding["entity_phrases"],
            application_context=resolved_application_context,
            step_context = step_context,
            judge_context = judge_context,
            step_data=step_data
        )

        if not program_data:
            program_data = {}

        data_details = {}
        data_details.update(program_data)
        data_details["strategy"] = "by_program"
        data_details["linked_entities"] = data_details['results']
        del data_details['results']

        return {
            "question": normalized_question,
            "original_question": overarching_question,
            "grounding": grounding,
            **data_details
        }

    def select_initial_data(self,
        question: str,
        schema_context: str,
        original_question: Optional[str] = None,
        application_context: Optional[str] = None,
        ):

        resolved_application_context = self.resolve_application_context(
            application_context
        )

        overarching_question = str(original_question or question).strip() or question
        normalized_question = question.strip() or overarching_question
        direct_uri_data = self.initial_data_from_explicit_uri(
            question=question,
            original_question=overarching_question,
        )
        if direct_uri_data is not None:
            return direct_uri_data

        grounding = self.ground_synthetic_questions(
            question=normalized_question,
            schema_context=schema_context,
            application_context=resolved_application_context,
        )



        program_data = self.initial_retrieval_strategy(
            question=normalized_question,
            schema_context=schema_context,
            candidate_classes=grounding["candidate_classes"],
            entity_phrases=grounding["entity_phrases"],
            candidate_relations=grounding["candidate_relations"],
            application_context=resolved_application_context,
        )

        data_details = {}
        if not program_data:
            linked_entities = self.object_db.link_entities_from_phrases(
                phrases=clean_string_list(
                    grounding["entity_phrases"] + [normalized_question]
                ),
                class_hints=grounding["candidate_classes"],
                limit=5,
            )

            data_details["strategy"] = "by_linked_data"
            data_details["linked_entities"] = linked_entities
        else:
            if len(program_data["results"]) == 1 and len(program_data["results"]['set_1']) == 0:
                linked_entities = self.object_db.link_entities_from_phrases(
                    phrases=clean_string_list(
                        grounding["entity_phrases"] + [normalized_question]
                    ),
                    class_hints=grounding["candidate_classes"],
                    limit=5,
                )
                data_details["strategy"] = "by_linked_data"
                data_details["linked_entities"] = linked_entities
            else:
                data_details.update(program_data)
                data_details["strategy"] = "by_program"
                data_details["linked_entities"] = data_details['results']
                del data_details['results']



        return {
            "question": normalized_question,
            "original_question": overarching_question,
            "grounding": grounding,
            **data_details
        }

    def summarize_answer_from_execution(
        self,
        original_question: str,
        current_question: str,
        schema_context: str,
        # steps: List[Dict[str, Any]],
        schema_reasoning: Optional[Dict[str, Any]] = None,
        step_context: str = "",
        application_context: Optional[str] = None,
    ) -> str:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        evidence_blocks = [
            self.format_schema_reasoning(schema_reasoning or {}),
        #    self.format_step_results(steps),
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
        step_context: str,
        # latest_step_results: str,
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
                step_context=step_context,
                # latest_step_results=latest_step_results,
                partial_answer=partial_answer,
                judge_feedback=judge_feedback,
            )
        next_question = str(getattr(prediction, "next_question", "")).strip()

        if not next_question:
            latest_focus = judge_feedback.strip() #or latest_step_results.strip()
            if latest_focus:
                next_question = (
                    f"Given the current evidence, what should be retrieved next to answer "
                    f"'{original_question}'? Focus on: {latest_focus}"
                )
            else:
                next_question = original_question

        return {"next_question": next_question}

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
                schema_context=schema_context,
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
        schema_context:str,
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
            schema_context=schema_context,
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

    def ground_synthetic_questions_path(
        self,
        question:str,
        schema_context:str,
        application_context:str,
        step_context:str,
        judge_context:str,
        )-> Dict[str, Any]:

        resolved_application_context = self.resolve_application_context(
            application_context
        )

        metadata = self.synthetic_question_retriever.get_programs_path_metadata()
        available_classes = [x["start_node"] for x in metadata]

        with dspy.context(lm=self.llm.llm):
            prediction = self.synthetic_question_path_grounding_predictor(
                question=question,
                application_context=resolved_application_context,
                schema_context=schema_context,
                step_context=step_context,
                judge_context =judge_context,
                available_classes=available_classes,
            )

        raw_candidate_classes = clean_string_list(
            getattr(prediction, "candidate_classes", [])
        )
        entity_phrases = clean_string_list(
            getattr(prediction, "entity_phrases", [])
        )

        available_class_set = set(available_classes)
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

        __entity_phrases = self.extract_literal_phrases(question)
        entity_phrases.extend(__entity_phrases)


        return {
            "candidate_classes": candidate_classes,
            "entity_phrases": entity_phrases,
            "candidate_relations": [],
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

        metadata = self.synthetic_question_retriever.get_programs_from_prop_metadata()
        available_classes = [x["class"] for x in metadata]
        available_relations = [x["relation"] for x in metadata]

        with dspy.context(lm=self.llm.llm):
            prediction = self.synthetic_question_grounding_predictor(
                question=question,
                application_context=resolved_application_context,
                schema_context=schema_context,
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



    @staticmethod
    def is_stringified_list(value: str) -> bool:
        """
        Detect if a string represents a Python list.

        Args:
            value: Input string

        Returns:
            bool: True if it can be safely parsed as a list
        """
        if not isinstance(value, str):
            return False

        value = value.strip()

        # Quick structural check (fast fail)
        if not (value.startswith("[") and value.endswith("]")):
            return False

        try:
            parsed: Any = ast.literal_eval(value)
            return isinstance(parsed, list)
        except (ValueError, SyntaxError):
            return False

    @staticmethod
    def parse_stringified_list(value: str) -> List[str]:
        """
        Convert a stringified list into a Python list.

        Args:
            value: Input string

        Returns:
            List[str]: Parsed list

        Raises:
            ValueError: If input is not a valid stringified list
        """
        parsed = ast.literal_eval(value)

        # Optional: enforce string elements
        if not all(isinstance(x, str) for x in parsed):
            _temp = []
            for p in parsed:
                _temp.append(str(p))

            parsed = _temp

        return parsed

    @staticmethod
    def is_stringified_dict(value: str) -> bool:
        if not isinstance(value, str):
            return False

        value = value.strip()
        if not (value.startswith("{") and value.endswith("}")):
            return False

        try:
            parsed: Any = ast.literal_eval(value)
            return isinstance(parsed, dict)
        except (ValueError, SyntaxError):
            return False

    @staticmethod
    def parse_stringified_dict(value: str) -> Dict[str, str]:
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, dict):
            raise ValueError("Input is not a valid stringified dict.")

        return {
            str(key): str(item)
            for key, item in parsed.items()
        }

    def synthetic_question_path_run(
        self,
        step_num:int,
        question:str,
        schema_context:str,
        candidate_classes:List[str],
        entity_phrases:List[str],
        application_context:str,
        step_context:str,
        judge_context:str,
        step_data:List[Dict[str,Any]],
        ) -> Optional[Dict[str,Any]]:

        resolved_application_context = self.resolve_application_context(
            application_context
        )

        objects_data_function = self.synthetic_question_retriever.get_attr_of_object()

        function_rows = self.synthetic_question_retriever.get_path_questions(
                start_nodes=candidate_classes,
                end_nodes=candidate_classes
            )

        functions = [f'{x["program_id"]}=>{x["solves"]}' for x in function_rows]


        functions.append(
            f'{objects_data_function["program_id"]}=>'
            f'{objects_data_function["solves"]}'
        )

        with dspy.context(lm=self.llm.llm):
            prediction = self.select_synthetic_question_predictor(
                question=question,
                application_context=resolved_application_context,
                schema_context=schema_context,
                step_context=step_context,
                judge_context=judge_context,
                functions = functions,
                entity_phrases=entity_phrases,
            )

        selected_function_id = str(
            getattr(prediction, "function_id", "")
        ).strip().strip("\"'")

        if "=>" in selected_function_id:
            selected_function_id = selected_function_id.split("=>")[0]


        program:Optional[Dict[str,Any]] = self.synthetic_question_retriever.get_program_by_id(
            selected_function_id
        )
        if not program:
            for function in functions:
                candidate_program_id = function.split("=>", 1)[0].strip()
                if (
                    selected_function_id == candidate_program_id
                    or selected_function_id in candidate_program_id
                    or candidate_program_id in selected_function_id
                ):
                    program = self.synthetic_question_retriever.get_program_by_id(
                        candidate_program_id
                    )
                    break
        
        if not program:

            return {
                "program_id": None,
                "parameter_values": [],
                "results": {"set_1":[]}
            }

        selected_step = self.build_selected_step_from_row(
            row=program,
            question=question,
            step_count=step_num,
            candidate_classes=candidate_classes,
            candidate_relations=[],
            previous_steps=step_data,
        )
        placeholders = self.extract_query_placeholders(program.get("code") or "")
        if not placeholders:
            placeholders = list(
                self.parse_serialized_dict(program.get("input_spec")).keys()
            )
        selected_step.setdefault("input_bindings", {})

        predecessor_context = "\n\n".join([
            f"{k}:\n{v}" for k,v in {
                "Previous Context":step_context,
                "Answer and Feedback":judge_context,
                "program Details": f"Obj Class {program['start_node']}"
                }.items()
        ])

        parameter_sets = self.resolve_step_parameters(
            question=question,
            predecessor_context=predecessor_context,
            schema_context=schema_context,
            step=selected_step,
            row= program,
            previous_steps=step_data,
            application_context=resolved_application_context,
        )

        if parameter_sets:
            extended = []
            for ps in parameter_sets:
                for k,v in ps.items():
                    if DependencyGraphRuntime.is_stringified_list(v):
                        vals = DependencyGraphRuntime.parse_stringified_list(v)
                        for vls in vals:
                            if DependencyGraphRuntime.is_stringified_dict(vls):
                                vals2 = DependencyGraphRuntime.parse_stringified_dict(vls)
                                if "value" in vals and len(vals) == 1:
                                    extended.append(
                                        {k: vals2["value"]}
                                    )
                                else:
                                    for dict_key, dict_value in vals2.items():
                                        extended.append(
                                            {k:dict_value}
                                        )
                            else:
                                extended.append(
                                    {k:vls}
                                )

                    elif DependencyGraphRuntime.is_stringified_dict(v):
                        vals = DependencyGraphRuntime.parse_stringified_dict(v)
                        for dict_key, dict_value in vals.items():
                            extended.append(
                                {k:dict_value}
                            )
                    else:
                        extended.append(
                                {k:v}
                            )

            parameter_sets = extended

            _temp_list = []
            for ps in parameter_sets:
                _temp = {}
                for k,v in ps.items():
                    if k not in ['class_uri', 'obj_uri', 'obj', "prop_value"]:
                        k = 'obj'
                    try:
                        _temp[k] = self.graph_manager.resolve_curie(v)
                    except TypeError as te:
                        print(v)

                _temp_list.append(_temp)

            parameter_sets = _temp_list



        if not parameter_sets:
            fallback_parameter_set: Dict[str, str] = {}
            for placeholder in placeholders:
                if placeholder == "class_uri" and candidate_classes:
                    fallback_parameter_set[placeholder] = candidate_classes[0]
                elif placeholder in {"obj", "obj_uri"} and entity_phrases:
                    fallback_parameter_set[placeholder] = entity_phrases[0]
                elif placeholder == "prop_value":
                    fallback_parameter_set[placeholder] = (
                        entity_phrases[0] if entity_phrases else question
                    )
            parameter_sets = self.normalize_parameter_sets(
                [fallback_parameter_set],
                placeholders=placeholders,
            )

        query_results: Dict[str, List[Dict[str, Any]]] = {}

        for index, parameter_values in enumerate(parameter_sets, start=1):
            rendered_query = regex_add_strings(program["code"], **parameter_values)
            results_df = self.graph_manager.query(
                rendered_query,
                add_header_tail=self.query_needs_header(rendered_query),
                resolve_curie=True,
            )
            query_results[f"set_{index}"] = results_df.to_dict("records")

        if not query_results:
            query_results["set_1"] = []

        return {
            "program_id": program["program_id"],
            "parameter_values": parameter_sets,
            "results": query_results
        }

    @staticmethod
    def extract_quoted_phrases(question: str) -> List[str]:
        double_quoted = re.findall(r'"([^"]+)"', question)
        double_backtick_quoted = re.findall(r"``([^`]+)``", question)
        single_backtick_quoted = re.findall(r"(?<!`)`([^`]+)`(?!`)", question)
        return clean_string_list(
            double_quoted + double_backtick_quoted + single_backtick_quoted
        )

    def initial_retrieval_strategy(
        self,
        question: str,
        schema_context: str,
        candidate_classes: List[str],
        candidate_relations:List[str],
        entity_phrases: List[str],
        application_context: Optional[str] = None,):

        resolved_application_context = self.resolve_application_context(
            application_context
        )

        all_objects_function = self.synthetic_question_retriever.get_objects_of_a_class()

        function_rows = self.synthetic_question_retriever.get_programs_from_prop(
                class_candidates=candidate_classes,
                relation= candidate_relations
            )

        functions = [f'{x["program_id"]}=>{x["solves"]}' for x in function_rows]
        if all_objects_function:
            functions.append(
                f'{all_objects_function["program_id"]}=>'
                f'{all_objects_function["solves"]}'
            )
        if not functions:
            return None

        with dspy.context(lm=self.llm.llm):
            prediction = self.initial_data_synthetic_question_predictor(
                question=question,
                application_context=resolved_application_context,
                schema_context=schema_context,
                functions = functions,
                entity_phrases=entity_phrases,
            )

        selected_function_id = str(
            getattr(prediction, "function_id", "")
        ).strip().strip("\"'")

        if "=>" in selected_function_id:
            selected_function_id = selected_function_id.split("=>")[0]

        program:Optional[Dict[str,Any]] = self.synthetic_question_retriever.get_program_by_id(
            selected_function_id
        )
        if not program:
            for function in functions:
                candidate_program_id = function.split("=>", 1)[0].strip()
                if (
                    selected_function_id == candidate_program_id
                    or selected_function_id in candidate_program_id
                    or candidate_program_id in selected_function_id
                ):
                    program = self.synthetic_question_retriever.get_program_by_id(
                        candidate_program_id
                    )
                    break

        if not program:
            return None

        selected_step = self.build_selected_step_from_row(
            row=program,
            question=question,
            step_count=0,
            candidate_classes=candidate_classes,
            candidate_relations=candidate_relations,
            previous_steps=[],
        )
        placeholders = self.extract_query_placeholders(program.get("code") or "")
        if not placeholders:
            placeholders = list(
                self.parse_serialized_dict(program.get("input_spec")).keys()
            )
        selected_step.setdefault("input_bindings", {})
        if "prop_value" in placeholders and not selected_step["input_bindings"].get(
            "prop_value"
        ):
            selected_step["input_bindings"]["prop_value"] = (
                entity_phrases[0] if entity_phrases else question
            )

        parameter_sets = self.resolve_step_parameters(
            question=question,
            predecessor_context="",
            schema_context=schema_context,
            step=selected_step,
            row= program,
            previous_steps=[],
            application_context=resolved_application_context,
        )
        if not parameter_sets:
            fallback_parameter_set: Dict[str, str] = {}
            for placeholder in placeholders:
                if placeholder == "class_uri" and candidate_classes:
                    fallback_parameter_set[placeholder] = candidate_classes[0]
                elif placeholder in {"obj", "obj_uri"} and entity_phrases:
                    fallback_parameter_set[placeholder] = entity_phrases[0]
                elif placeholder == "prop_uri" and candidate_relations:
                    fallback_parameter_set[placeholder] = candidate_relations[0]
                elif placeholder == "prop_value":
                    fallback_parameter_set[placeholder] = (
                        entity_phrases[0] if entity_phrases else question
                    )
            parameter_sets = self.normalize_parameter_sets(
                [fallback_parameter_set],
                placeholders=placeholders,
            )

        if parameter_sets:
            extended = []
            for ps in parameter_sets:
                for k,v in ps.items():
                    if DependencyGraphRuntime.is_stringified_list(v):
                        vals = DependencyGraphRuntime.parse_stringified_list(v)
                        for vls in vals:
                            if DependencyGraphRuntime.is_stringified_dict(vls):
                                vals2 = DependencyGraphRuntime.parse_stringified_dict(vls)
                                if "value" in vals and len(vals) == 1:
                                    extended.append(
                                        {k: vals2["value"]}
                                    )
                                else:
                                    for dict_key, dict_value in vals2.items():
                                        extended.append(
                                            {k:dict_value}
                                        )
                            else:
                                extended.append(
                                    {k:vls}
                                )

                    elif DependencyGraphRuntime.is_stringified_dict(v):
                        vals = DependencyGraphRuntime.parse_stringified_dict(v)
                        for dict_key, dict_value in vals.items():
                            extended.append(
                                {k:dict_value}
                            )
                    else:
                        extended.append(
                                {k:v}
                            )

            parameter_sets = extended

            _temp_list = []
            for ps in parameter_sets:
                _temp = {}
                for k,v in ps.items():
                    if k not in ['class_uri', 'obj_uri', 'obj', "prop_value"]:
                        k = 'obj'

                    if k == "prop_value" and len(v)> 80:
                        v = self.extract_quoted_phrases(v)
                        if len(v)>0:
                            v = v[0]
                        else:
                            v = v[:80]

                    try:
                        _temp[k] = self.graph_manager.resolve_curie(v)
                    except TypeError as te:
                        print(v)

                _temp_list.append(_temp)

            parameter_sets = _temp_list

        query_results: Dict[str, List[Dict[str, Any]]] = {}

        for index, parameter_values in enumerate(parameter_sets, start=1):
            rendered_query = regex_add_strings(program["code"], **parameter_values)
            results_df = self.graph_manager.query(
                rendered_query,
                add_header_tail=self.query_needs_header(rendered_query),
                resolve_curie=True,
            )
            query_results[f"set_{index}"] = results_df.to_dict("records")

        if not query_results:
            query_results["set_1"] = []

        return {
            "program_id": program["program_id"],
            "parameter_values": parameter_sets,
            "results": query_results
        }

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
            elif placeholder == "prop_value":
                input_bindings[placeholder] = DependencyGraphRuntime.extract_quoted_phrases(question)

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

    def resolve_step_parameters(
        self,
        question: str,
        predecessor_context: str,
        schema_context:str,
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
        
        for k,v in candidate_parameter_values.items():
            candidate_parameter_values[k] = list(set(v))

        with dspy.context(lm=self.llm.llm):
            prediction = self.synthetic_question_parameter_predictor(
                question=question,
                application_context=resolved_application_context,
                schema_context = schema_context,
                predecessor_context = predecessor_context,
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
                    # candidates[placeholder].extend(
                    #     previous_step.get("important_entities", [])
                    # )
                    
                    if 'extracted_results' in previous_step:
                        for v in previous_step['extracted_results']:
                            candidates[placeholder].append(
                                        v['uri']
                                    )
                            for v1 in v['attributes']:
                                if  '-' != v1["object_class"]:
                                    candidates[placeholder].append(
                                        v1["object"]
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

            if placeholder in {"prop_uri", "prop_value"}:
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
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_application_context = self.resolve_application_context(
            application_context
        )
        current_answer = answer.strip()

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


        return {
            "answer": current_answer,
            "answered": answered,
            "feedback": feedback,
        }
