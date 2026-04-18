from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.FullContextAnswer import FullContextAnswerBaseline


class FakeStructuredLLM:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def structured_generate(self, prompt: str, structure: Any, system_prompt: str = "") -> dict[str, Any]:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, "structure": structure})
        return self.payload


class FakeRawLLM:
    def __init__(self, raw_response: str) -> None:
        self.raw_response = raw_response
        self.calls: list[dict[str, Any]] = []

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return self.raw_response


def test_full_context_baseline_returns_resolved_entities_and_evidence() -> None:
    llm = FakeStructuredLLM(
        {
            "answer": "The generated answer was produced by the LLM Chat execution and then used by Information Extractor.",
            "relevant_entities": [
                "ChatBS-NexGen:Data-id_20260413053510_702-generated_answer",
                "ChatBS-NexGen:llm_chat",
                "ChatBS-NexGen:information_extractor",
            ],
            "evidence": [
                {
                    "subject": "ChatBS-NexGen:Data-id_20260413053510_702-generated_answer",
                    "predicate": "prov:wasGeneratedBy",
                    "object": "ChatBS-NexGen:id_20260413053510_114",
                },
                {
                    "subject": "ChatBS-NexGen:id_20260413053513_620",
                    "predicate": "prov:used",
                    "object": "ChatBS-NexGen:Data-id_20260413053510_702-generated_answer",
                },
            ],
        }
    )
    baseline = FullContextAnswerBaseline(
        kg_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample.ttl",
        ontology_path=REPO_ROOT / "schema/WorkFlow.ttl",
        schema_json_path=REPO_ROOT / "schema/schemaV2.json",
        metadata_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample_metadata.json",
        application_description="A program to analyze and generate provenance data for ChatBS-NexGen.",
        llm=llm,
    )

    result = baseline.request("What happened to the generated answer?")

    assert "generated answer" in result["answer"].lower()
    assert any(entity["label"] == "generated answer" for entity in result["relevant_entities"])
    assert any(entity["label"] == "LLM Chat" for entity in result["relevant_entities"])
    assert any(entity["label"] == "Information Extractor" for entity in result["relevant_entities"])
    assert any(triple["predicate_id"] == "prov:wasGeneratedBy" for triple in result["evidence"])
    assert any(triple["predicate_id"] == "prov:used" for triple in result["evidence"])


def test_prompt_contains_application_description_kg_and_question() -> None:
    llm = FakeStructuredLLM(
        {
            "answer": "The LLM Chat execution used the user prompt and system prompt.",
            "relevant_entities": ["ChatBS-NexGen:llm_chat"],
            "evidence": [],
        }
    )
    baseline = FullContextAnswerBaseline(
        kg_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample.ttl",
        ontology_path=REPO_ROOT / "schema/WorkFlow.ttl",
        schema_json_path=REPO_ROOT / "schema/schemaV2.json",
        metadata_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample_metadata.json",
        application_description="A program to analyze and generate provenance data for ChatBS-NexGen.",
        llm=llm,
    )

    baseline.request("What inputs were used by the LLM Chat execution?")
    prompt = llm.calls[0]["prompt"]

    assert "A program to analyze and generate provenance data for ChatBS-NexGen." in prompt
    assert "Full execution KG:" in prompt
    assert "generated answer" in prompt
    assert "Question:\nWhat inputs were used by the LLM Chat execution?" in prompt


def test_raw_json_fallback_is_parsed() -> None:
    llm = FakeRawLLM(
        """
```json
{
  "answer": "The prompt was handled by gpt-4o using the system prompt and user prompt.",
  "relevant_entities": ["ChatBS-NexGen:LLM-id_20260413053510_114", "ChatBS-NexGen:llm_chat"],
  "evidence": [
    {
      "subject": "ChatBS-NexGen:LLM-id_20260413053510_114",
      "predicate": "workflow:llm_model",
      "object": "gpt-4o"
    }
  ]
}
```
""".strip()
    )
    baseline = FullContextAnswerBaseline(
        kg_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample.ttl",
        ontology_path=REPO_ROOT / "schema/WorkFlow.ttl",
        schema_json_path=REPO_ROOT / "schema/schemaV2.json",
        metadata_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample_metadata.json",
        llm=llm,
    )

    result = baseline.request("Which model handled the prompt?")

    assert "gpt-4o" in result["answer"]
    assert result["relevant_entities"]
    assert result["evidence"]
