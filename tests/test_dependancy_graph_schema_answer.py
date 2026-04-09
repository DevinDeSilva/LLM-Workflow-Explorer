import sys
from pathlib import Path


src_path = Path(__file__).resolve().parent.parent

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


from src.explainer.dependancy_graph_node import QuestionNode  # noqa: E402


class SchemaOnlyRuntime:
    def format_predecessor_context(self, predecessor_info):
        return ""

    def resolve_application_context(self, application_context=None):
        return (application_context or "").strip()

    def answer_question_from_schema(
        self,
        question,
        schema_context,
        max_tries=5,
        application_context=None,
    ):
        return {
            "answer": "The schema defines provone:Program as the workflow program class.",
            "answered": True,
            "draft_answer": "The schema defines provone:Program as the workflow program class.",
            "history": [
                {
                    "answered": True,
                    "feedback": "",
                    "answer": "The schema defines provone:Program as the workflow program class.",
                }
            ],
            "relevant_schema_info": [
                "provone:Program: workflow program class.",
            ],
        }

    def plan_synthetic_question_execution(self, *args, **kwargs):
        raise AssertionError("Schema-only answers should skip synthetic planning.")

    def execute_synthetic_question_plan(self, *args, **kwargs):
        raise AssertionError("Schema-only answers should skip synthetic execution.")


class SchemaFallbackRuntime:
    def __init__(self):
        self.judge_inputs = None

    def format_predecessor_context(self, predecessor_info):
        return ""

    def resolve_application_context(self, application_context=None):
        return (application_context or "").strip()

    def answer_question_from_schema(
        self,
        question,
        schema_context,
        max_tries=5,
        application_context=None,
    ):
        return {
            "answer": "The schema shows input and output ports, but not any concrete program instances.",
            "answered": False,
            "draft_answer": "The schema shows input and output ports, but not any concrete program instances.",
            "history": [
                {
                    "answered": False,
                    "feedback": "Instance-level evidence is still needed.",
                    "answer": "The schema shows input and output ports, but not any concrete program instances.",
                }
            ],
            "relevant_schema_info": [
                "provone:hasInPort links provone:Program to provone:Port.",
                "provone:hasOutPort links provone:Program to provone:Port.",
            ],
        }

    def plan_synthetic_question_execution(
        self,
        question,
        schema_context,
        predecessor_context="",
        application_context=None,
    ):
        return {
            "grounding": {},
            "candidate_synthetic_questions": [],
            "plan": [],
        }

    def execute_synthetic_question_plan(
        self,
        question,
        predecessor_context,
        plan,
        application_context=None,
    ):
        return {
            "steps": [],
            "answer": "",
        }

    def format_schema_reasoning(self, schema_reasoning):
        return "\n".join(
            ["Schema details:"]
            + [
                f"- {schema_fact}"
                for schema_fact in schema_reasoning.get("relevant_schema_info", [])
            ]
        )

    def format_step_results(self, steps):
        return ""

    def ensure_answer_quality(
        self,
        question,
        answer,
        evidence_context,
        max_tries=5,
        application_context=None,
    ):
        self.judge_inputs = {
            "question": question,
            "answer": answer,
            "evidence_context": evidence_context,
            "max_tries": max_tries,
            "application_context": application_context,
        }
        return {
            "answer": "Programs expose ports in the schema, and concrete instances were not retrieved.",
            "answered": True,
            "history": [
                {
                    "answered": True,
                    "feedback": "",
                    "answer": "Programs expose ports in the schema, and concrete instances were not retrieved.",
                }
            ],
        }


def test_question_node_returns_schema_only_answer():
    runtime = SchemaOnlyRuntime()
    node = QuestionNode(id="1", question="What is provone:Program?")

    result = node.solve(
        schema_info="Workflow schema",
        predecessor_info={},
        runtime=runtime,
    )

    assert result["answer"] == "The schema defines provone:Program as the workflow program class."
    assert result["judge"] == result["schema_reasoning"]["judge"]
    assert result["schema_reasoning"]["relevant_schema_info"] == [
        "provone:Program: workflow program class.",
    ]


def test_question_node_carries_schema_reasoning_into_final_judge():
    runtime = SchemaFallbackRuntime()
    node = QuestionNode(id="1", question="Which program ports are defined and used?")

    result = node.solve(
        schema_info="Workflow schema",
        predecessor_info={},
        runtime=runtime,
    )

    assert result["answer"] == "Programs expose ports in the schema, and concrete instances were not retrieved."
    assert runtime.judge_inputs is not None
    assert runtime.judge_inputs["answer"] == (
        "The schema shows input and output ports, but not any concrete program instances."
    )
    assert runtime.judge_inputs["application_context"] == ""
    assert "Schema details:" in runtime.judge_inputs["evidence_context"]
    assert "provone:hasInPort links provone:Program to provone:Port." in runtime.judge_inputs["evidence_context"]
