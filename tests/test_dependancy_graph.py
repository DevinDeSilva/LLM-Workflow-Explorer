import sys
from pathlib import Path
from types import SimpleNamespace

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
