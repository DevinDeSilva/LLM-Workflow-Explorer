import sys
from pathlib import Path
from types import SimpleNamespace
from types import MethodType

src_path = Path(__file__).resolve().parent.parent

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import src.explainer.dependancy_graph as dependency_graph_module
from src.config.experiment import ApplicationInfo
from src.explainer.dependancy_graph import DependancyGraph
from src.templates.demos.dependancy_graph import (
    build_information_required_fewshot_examples,
)
from src.utils.utils import clean_string_list


class DummyContext:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_information_required_uses_dspy_predict_and_returns_list(monkeypatch):
    captured = {}

    class FakeInformationRequiredPredictor:
        def __init__(self):
            self.demos = []

        def __call__(self, **kwargs):
            captured["information_required_inputs"] = kwargs
            return SimpleNamespace(
                sub_questions=[
                    "  the relevant workflow task  ",
                    "the model used by that task",
                    "the relevant workflow task",
                ]
            )

    class FakeFilterPredictor:
        def __call__(self, **kwargs):
            captured["filter_inputs"] = kwargs
            return SimpleNamespace(filtered_sub_question=[])

    monkeypatch.setattr(dependency_graph_module.dspy, "context", lambda **kwargs: DummyContext(**kwargs))

    graph = DependancyGraph.__new__(DependancyGraph)
    graph.app_info = ApplicationInfo(description="An application for testing LLM prompt workflows.")
    graph.llm = SimpleNamespace(llm="fake-dspy-lm")
    graph.information_required_predictor = FakeInformationRequiredPredictor()
    graph.filter_sub_question_predictor = FakeFilterPredictor()

    result = graph.information_required(
        "Which model handled the prompt?",
        "Workflow schema\nClasses:\n- workflow:Task",
    )

    assert result == [
        "the relevant workflow task",
        "the model used by that task",
    ]
    assert captured["information_required_inputs"] == {
        "user_query": "Which model handled the prompt?",
        "schema_context": "Workflow schema\nClasses:\n- workflow:Task",
        "application_context": "An application for testing LLM prompt workflows.",
    }
    assert captured["filter_inputs"] == {
        "original_question": "Which model handled the prompt?",
        "application_context": "An application for testing LLM prompt workflows.",
        "sub_questions": [
            "0). the relevant workflow task",
            "1). the model used by that task",
        ],
    }


def test_build_information_required_fewshot_examples_contains_placeholders():
    examples = build_information_required_fewshot_examples()

    assert len(examples) == 2
    assert examples[0].user_query == "<ADD_USER_QUERY_EXAMPLE_1>"
    assert examples[0].application_context == "Application context will be supplied at runtime."
    assert examples[0].information_required == [
        "<ADD_INFORMATION_REQUIRED_ITEM_1>",
        "<ADD_INFORMATION_REQUIRED_ITEM_2>",
    ]
    assert examples[1].user_query == "<ADD_USER_QUERY_EXAMPLE_2>"
    assert examples[1].application_context == "Application context will be supplied at runtime."
    assert examples[1].information_required == [
        "<ADD_INFORMATION_REQUIRED_ITEM_1>",
        "<ADD_INFORMATION_REQUIRED_ITEM_2>",
        "<ADD_INFORMATION_REQUIRED_ITEM_3>",
    ]


def test_clean_string_list_normalizes_and_deduplicates_items():
    assert clean_string_list([
        "  the execution being asked about  ",
        "the program associated with that execution",
        "the execution being asked about",
        "",
    ]) == [
        "the execution being asked about",
        "the program associated with that execution",
    ]


def test_plan_synthetic_question_execution_uses_sqretriever_rows_and_filters_invalid_programs(monkeypatch):
    captured = {}

    class FakeGroundingPredictor:
        def __call__(self, **kwargs):
            captured["grounding_inputs"] = kwargs
            return SimpleNamespace(
                candidate_classes=["provone:Execution"],
                candidate_relations=["dcterms:identifier"],
                entity_phrases=["execution 1_2"],
            )

    class FakePlanPredictor:
        def __call__(self, **kwargs):
            captured["plan_inputs"] = kwargs
            return SimpleNamespace(
                execution_plan_json="""
                [
                  {
                    "step_id": "step1",
                    "sub_question": "Find the execution",
                    "program_id": "valid-program",
                    "input_bindings": {
                      "obj_uri": "execution 1_2"
                    },
                    "expected_classes": ["provone:Execution"]
                  },
                  {
                    "step_id": "step2",
                    "sub_question": "Ignore this step",
                    "program_id": "missing-program",
                    "input_bindings": {},
                    "expected_classes": []
                  }
                ]
                """
            )

    class FakeSQRetriever:
        def __init__(self):
            self.rows = [
                {
                    "program_id": "valid-program",
                    "category": "object-level|from-object",
                    "statement": "Find execution identifier values",
                    "start_node": None,
                    "end_node": None,
                    "focal_relation": "dcterms:identifier",
                    "focal_node": "provone:Execution",
                    "code": "SELECT ?value WHERE { <{obj_uri}> ?p ?value . }",
                    "input_spec": "{'obj_uri': 'Execution URI'}",
                },
                {
                    "program_id": "other-program",
                    "category": "path-level",
                    "statement": "Find related data",
                    "start_node": "provone:Execution",
                    "end_node": "provone:Data",
                    "focal_relation": None,
                    "focal_node": None,
                    "code": "SELECT ?value WHERE { <{obj}> ?p ?value . }",
                    "input_spec": "{'obj': 'Execution URI'}",
                },
            ]

        def retrieve_object_level_from_object(self, focal_node=None, focal_relation=None, **filters):
            return [
                row for row in self.rows
                if row["category"] == "object-level|from-object"
                and (focal_node is None or row["focal_node"] == focal_node)
                and (focal_relation is None or row["focal_relation"] == focal_relation)
            ]

        def retrieve_object_level_from_prop(self, focal_node=None, focal_relation=None, **filters):
            return []

        def retrieve_path_level(self, start_node=None, end_node=None, **filters):
            return [
                row for row in self.rows
                if row["category"] == "path-level"
                and (start_node is None or row["start_node"] == start_node)
                and (end_node is None or row["end_node"] == end_node)
            ]

    monkeypatch.setattr(dependency_graph_module.dspy, "context", lambda **kwargs: DummyContext(**kwargs))

    graph = DependancyGraph.__new__(DependancyGraph)
    graph.app_info = ApplicationInfo(description="An application for testing LLM prompt workflows.")
    graph.llm = SimpleNamespace(llm="fake-dspy-lm")
    graph.synthetic_question_retriever = FakeSQRetriever()
    graph.synthetic_questions_by_program_id = {
        row["program_id"]: row
        for row in graph.synthetic_question_retriever.rows
    }
    graph.synthetic_question_grounding_predictor = FakeGroundingPredictor()
    graph.synthetic_question_plan_predictor = FakePlanPredictor()

    result = graph.plan_synthetic_question_execution(
        question="Which execution has identifier 1_2?",
        schema_context="Workflow schema\nClasses:\n- provone:Execution",
        predecessor_context="",
    )

    assert [row["program_id"] for row in result["candidate_synthetic_questions"]] == [
        "valid-program",
        "other-program",
    ]
    assert result["plan"] == [
        {
            "step_id": "step1",
            "sub_question": "Find the execution",
            "program_id": "valid-program",
            "input_bindings": {"obj_uri": "execution 1_2"},
            "expected_classes": ["provone:Execution"],
        }
    ]
    assert captured["grounding_inputs"]["application_context"] == (
        "An application for testing LLM prompt workflows."
    )
    assert captured["plan_inputs"]["application_context"] == (
        "An application for testing LLM prompt workflows."
    )


def test_build_toplevel_dependancy_graph_passes_application_context(monkeypatch):
    captured = {}

    class FakeTopologyPredictor:
        def __call__(self, **kwargs):
            captured["topology_inputs"] = kwargs
            return SimpleNamespace(topology_graph="1 -> Q")

    monkeypatch.setattr(
        dependency_graph_module.dspy,
        "context",
        lambda **kwargs: DummyContext(**kwargs),
    )

    graph = DependancyGraph.__new__(DependancyGraph)
    graph.app_info = ApplicationInfo(description="Application workflow description.")
    graph.llm = SimpleNamespace(llm="fake-dspy-lm")
    graph.vertices = {}
    graph.build_topology_graph_predictor = FakeTopologyPredictor()

    graph.build_toplevel_dependancy_graph(
        "Which model handled the prompt?",
        [dependency_graph_module.QuestionNode(id="1", question="Find the execution")],
    )

    assert captured["topology_inputs"] == {
        "original_question": "Which model handled the prompt?",
        "application_context": "Application workflow description.",
        "sub_questions": ["1. Find the execution"],
    }


def test_process_dependancy_graph_solves_root_node():
    graph = DependancyGraph.__new__(DependancyGraph)
    graph.app_info = ApplicationInfo(description="Workflow application description.")
    graph.vertices = {
        "0": dependency_graph_module.QuestionNode(
            id="0",
            question="Which model handled the prompt?",
        )
    }
    graph.adjacency_matrix = [[0]]

    graph.format_predecessor_context = MethodType(lambda self, predecessor_info: "", graph)
    graph.answer_question_from_schema = MethodType(
        lambda self, question, schema_context, max_tries=2, application_context=None: {
            "answer": "",
            "answered": False,
            "draft_answer": "",
            "history": [],
            "relevant_schema_info": [],
        },
        graph,
    )
    graph.plan_synthetic_question_execution = MethodType(
        lambda self, question, schema_context, predecessor_context="", application_context=None: {
            "grounding": {"candidate_classes": ["workflow:Large_Language_Model_Output"]},
            "candidate_synthetic_questions": [{"program_id": "valid-program"}],
            "plan": [
                {
                    "step_id": "step1",
                    "sub_question": "Find the execution",
                    "program_id": "valid-program",
                    "input_bindings": {},
                    "expected_classes": ["provone:Execution"],
                }
            ],
        },
        graph,
    )
    graph.execute_synthetic_question_plan = MethodType(
        lambda self, question, predecessor_context, plan, application_context=None: {
            "steps": [
                {
                    "step_id": "step1",
                    "sub_question": "Find the execution",
                    "program_id": "valid-program",
                    "important_entities": ["ex:Execution1"],
                    "results": {"set_1": [{"value": "ex:Execution1"}]},
                    "answer": "The relevant execution is ex:Execution1.",
                }
            ],
            "answer": "The prompt was handled by gpt-4o.",
        },
        graph,
    )
    graph.format_step_results = MethodType(lambda self, steps: "step results", graph)
    graph.ensure_answer_quality = MethodType(
        lambda self, question, answer, evidence_context, max_tries=2, application_context=None: {
            "answer": answer,
            "answered": True,
            "history": [{"answered": True, "feedback": "", "answer": answer}],
        },
        graph,
    )

    result = graph.process_dependancy_graph(schema_context="Workflow schema")

    assert result["id"] == "0"
    assert result["answer"] == "The prompt was handled by gpt-4o."
    assert result["synthetic_questions_plan"] == [
        {
            "step_id": "step1",
            "sub_question": "Find the execution",
            "program_id": "valid-program",
            "input_bindings": {},
            "expected_classes": ["provone:Execution"],
        }
    ]
    assert result["retrieved_objects"] == ["ex:Execution1"]
