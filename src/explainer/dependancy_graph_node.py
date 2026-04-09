from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.explainer.dependancy_graph_runtime import DependencyGraphRuntimeBase


class QuestionNode(BaseModel):
    id: str
    question: str
    node_type: Optional[str] = None

    def solve(
        self,
        schema_info: str,
        predecessor_info: Dict[str, Any],
        runtime: Optional["DependencyGraphRuntimeBase"] = None,
        max_tries: int = 5,
        max_rounds: int = 5,
        application_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        if runtime is None:
            return {
                "answer": "",
                "plan": [],
                "execution": [],
                "judge": [],
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

            summarize_answer = getattr(
                runtime,
                "summarize_answer_from_execution",
                None,
            )
            if callable(summarize_answer):
                final_answer = str(
                    summarize_answer(
                        original_question=self.question,
                        current_question=executed_question,
                        schema_context=schema_info,
                        predecessor_context=predecessor_context,
                        schema_reasoning=solution["schema_reasoning"],
                        steps=solution["execution"],
                        step_context=step_context,
                        application_context=resolved_application_context,
                    )
                ).strip()
            else:
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

            if callable(getattr(runtime, "build_step_context", None)):
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
        runtime: "DependencyGraphRuntimeBase",
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
