import sys
from pathlib import Path
from types import SimpleNamespace

src_path = Path(__file__).resolve().parent.parent

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import src.explainer.dependancy_graph as dependency_graph_module
from src.config.experiment import ApplicationInfo
from src.explainer.dependancy_graph import DependancyGraphCreation


class DummyContext:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_information_required_uses_dspy_predict_and_returns_list(monkeypatch):
    captured = {}

    class FakePredictor:
        def __init__(self):
            self.demos = []
        
        def __call__(self, **kwargs):
            captured["inputs"] = kwargs
            return SimpleNamespace(
                information_required=[
                    "the relevant workflow task",
                    "the model used by that task",
                ]
            )

    monkeypatch.setattr(dependency_graph_module.dspy, "context", lambda **kwargs: DummyContext(**kwargs))

    graph = DependancyGraphCreation.__new__(DependancyGraphCreation)
    graph.app_info = ApplicationInfo(description="An application for testing LLM prompt workflows.")
    graph.llm = "fake-dspy-lm"
    graph.information_required_predictor = FakePredictor()

    result = graph.information_required(
        "Which model handled the prompt?",
        "Workflow schema\nClasses:\n- workflow:Task",
    )

    assert result == [
        "the relevant workflow task",
        "the model used by that task",
    ]
    assert captured["inputs"] == {
        "user_query": "Which model handled the prompt?",
        "schema_context": "Workflow schema\nClasses:\n- workflow:Task",
        "application_context": "An application for testing LLM prompt workflows.",
    }


def test_build_information_required_fewshot_examples_contains_placeholders():
    examples = DependancyGraphCreation.build_information_required_fewshot_examples()

    assert len(examples) == 2
    assert examples[0].user_query == "<ADD_USER_QUERY_EXAMPLE_1>"
    assert examples[0].information_required == [
        "<ADD_INFORMATION_REQUIRED_ITEM_1>",
        "<ADD_INFORMATION_REQUIRED_ITEM_2>",
    ]
    assert examples[1].user_query == "<ADD_USER_QUERY_EXAMPLE_2>"
    assert examples[1].information_required == [
        "<ADD_INFORMATION_REQUIRED_ITEM_1>",
        "<ADD_INFORMATION_REQUIRED_ITEM_2>",
        "<ADD_INFORMATION_REQUIRED_ITEM_3>",
    ]


def test_clean_requirement_list_normalizes_and_deduplicates_items():
    assert DependancyGraphCreation._clean_requirement_list([
        "  the execution being asked about  ",
        "the program associated with that execution",
        "the execution being asked about",
        "",
    ]) == [
        "the execution being asked about",
        "the program associated with that execution",
    ]
