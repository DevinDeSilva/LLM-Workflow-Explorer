from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.VectorSimilarityAnswer import VectorSimilarityAnswerBaseline


class FakeEmbeddings:
    terms = ["llm", "chat", "prompt", "generated", "answer", "extractor"]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        normalized = text.lower()
        return [float(normalized.count(term)) for term in self.terms]


class FakeStructuredLLM:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def structured_generate(
        self,
        prompt: str,
        structure: Any,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "structure": structure,
            }
        )
        return self.payload


def test_vector_similarity_baseline_retrieves_object_segments_and_answers() -> None:
    llm = FakeStructuredLLM(
        {
            "answer": "The LLM Chat execution used the system prompt and user prompt."
        }
    )
    baseline = VectorSimilarityAnswerBaseline(
        kg_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample.ttl",
        ontology_path=REPO_ROOT / "schema/WorkFlow.ttl",
        schema_json_path=REPO_ROOT / "schema/schemaV2.json",
        metadata_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample_metadata.json",
        llm=llm,
        embeddings_model=FakeEmbeddings(),
        top_k=5,
    )

    result = baseline.request("What inputs were used by the LLM Chat execution?")

    assert "system prompt" in result["answer"]
    assert len(result["retrieved_objects"]) == 5
    assert result["relevant_entities"]
    assert result["evidence"]
    assert result["token_usage"]["estimated"] is True
    assert result["token_usage"]["prompt_tokens"] > 0
    assert "Retrieved object descriptions:" in llm.calls[0]["prompt"]


def test_vector_similarity_search_returns_top_k_sorted_results() -> None:
    baseline = VectorSimilarityAnswerBaseline(
        kg_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample.ttl",
        ontology_path=REPO_ROOT / "schema/WorkFlow.ttl",
        schema_json_path=REPO_ROOT / "schema/schemaV2.json",
        metadata_path=REPO_ROOT / "usecases/chatbs/data/1_sample_graph/chatbs_sample_metadata.json",
        llm=FakeStructuredLLM({"answer": "ok"}),
        embeddings_model=FakeEmbeddings(),
        top_k=3,
    )

    results = baseline.search("generated answer", limit=3)

    assert len(results) == 3
    assert results[0].score >= results[1].score >= results[2].score
    assert any("generated answer" in result.description.lower() for result in results)
