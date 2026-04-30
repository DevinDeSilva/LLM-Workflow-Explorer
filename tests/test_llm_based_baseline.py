from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.LLMbased import GroundedWorkflowBaseline


class FakeAIMessage:
    content = ""
    usage_metadata = {"input_tokens": 21, "output_tokens": 8, "total_tokens": 29}
    response_metadata = {"model_name": "fake-rewrite-model"}
    additional_kwargs: dict[str, Any] = {}


class FakeStructuredRunnable:
    def __init__(self, payload: dict[str, Any], raw_response: Any) -> None:
        self.payload = payload
        self.raw_response = raw_response

    async def ainvoke(self, _messages: list[Any]) -> dict[str, Any]:
        return {"parsed": self.payload, "raw": self.raw_response}


class FakeChatModel:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.runnable = FakeStructuredRunnable(payload, FakeAIMessage())

    def with_structured_output(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> FakeStructuredRunnable:
        return self.runnable


class FakeLangChainLLM:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.llm = FakeChatModel(payload)


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
    assert any(entity["label"] == "llm chat 1 1" for entity in result["relevant_entities"])
    assert any(entity["label"] == "LLM Chat" for entity in result["relevant_entities"])
    assert result["token_usage"]["source"] == "none"


def test_generated_answer_question_returns_trace_summary(baseline: GroundedWorkflowBaseline) -> None:
    result = baseline.request("What happened to the generated answer?")

    assert "generated answer" in result["answer"].lower()
    assert "LLM Chat" in result["answer"]
    assert "Information Extractor" in result["answer"]
    assert any(entity["label"] == "generated answer" for entity in result["relevant_entities"])
    assert any(triple["predicate_id"] in {"prov:wasGeneratedBy", "prov:used"} for triple in result["evidence"])


def test_llm_based_baseline_records_rewrite_token_usage() -> None:
    llm = FakeLangChainLLM({"answer": "The LLM Chat execution used the prompt."})
    baseline = GroundedWorkflowBaseline(
        kg_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample.ttl",
        ontology_path=REPO_ROOT / "schema/WorkFlow.ttl",
        schema_json_path=REPO_ROOT / "schema/schemaV2.json",
        metadata_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample_metadata.json",
        llm=llm,
    )

    result = baseline.request("What inputs were used by the LLM Chat execution?")

    assert result["answer"].startswith("The LLM Chat execution used the prompt.")
    assert result["token_usage"]["estimated"] is False
    assert result["token_usage"]["source"] == "provider"
    assert result["token_usage"]["prompt_tokens"] == 21
    assert result["token_usage"]["completion_tokens"] == 8
    assert result["token_usage"]["total_tokens"] == 29
    assert "fake-rewrite-model" in result["token_usage"]["models"]
