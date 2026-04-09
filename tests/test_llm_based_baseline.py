from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.LLMbased import GroundedWorkflowBaseline


@pytest.fixture(scope="module")
def baseline() -> GroundedWorkflowBaseline:
    return GroundedWorkflowBaseline(
        kg_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample.ttl",
        ontology_path=REPO_ROOT / "schema/WorkFlow.ttl",
        schema_json_path=REPO_ROOT / "schema/schemaV2.json",
        metadata_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample_metadata.json",
    )


def test_model_question_returns_grounded_answer_and_entities(baseline: GroundedWorkflowBaseline) -> None:
    result = baseline.request("Which model handled the prompt?")

    assert "gpt-4o" in result["answer"]
    assert "system prompt" in result["answer"]
    assert "user prompt" in result["answer"]
    assert result["relevant_entities"]
    assert any("gpt-4o model" in entity["label"] for entity in result["relevant_entities"])
    assert result["evidence"]


def test_input_question_centers_on_llm_chat_execution(baseline: GroundedWorkflowBaseline) -> None:
    result = baseline.request("What inputs were used by the LLM Chat execution?")

    assert "LLM Chat" in result["answer"]
    assert "with inputs user prompt, system prompt" in result["answer"]
    assert any(entity["label"] == "execution 1_1" for entity in result["relevant_entities"])
    assert any(entity["label"] == "LLM Chat" for entity in result["relevant_entities"])


def test_generated_answer_question_returns_trace_summary(baseline: GroundedWorkflowBaseline) -> None:
    result = baseline.request("What happened to the generated answer?")

    assert "generated answer" in result["answer"].lower()
    assert "LLM Chat" in result["answer"]
    assert "Information Extractor" in result["answer"]
    assert any(entity["label"] == "generated answer" for entity in result["relevant_entities"])
    assert any(triple["predicate_id"] in {"prov:wasGeneratedBy", "prov:used"} for triple in result["evidence"])
