# VectorSimilarityAnswer Baseline

`VectorSimilarityAnswerBaseline` segments a TTL knowledge graph into one object
description per URI subject, embeds those object descriptions, retrieves the
top similar objects for a question, and answers using only those retrieved
object details.

It returns:

- `answer`: an LLM answer grounded in the retrieved object descriptions
- `relevant_entities`: the retrieved KG objects used as context
- `evidence`: object-property triples from the retrieved segments
- `retrieved_objects`: the top vector-similarity matches and scores
- `token_usage`: provider token counts when available, otherwise an estimate

## Usage

```python
from baselines.VectorSimilarityAnswer import VectorSimilarityAnswerBaseline

baseline = VectorSimilarityAnswerBaseline(
    kg_path="usecases/chatbs/data/1_sample_graph/chatbs_sample.ttl",
    ontology_path="schema/WorkFlow.ttl",
    schema_json_path="schema/schemaV2.json",
    metadata_path="usecases/chatbs/data/1_sample_graph/chatbs_sample_metadata.json",
    llm_type="lmstudio",
    llm_config={"model": "openai/gpt-oss-20b", "base_url": "http://localhost:1234/v1"},
    embedding_type="lmstudio",
    embedding_config={
        "model": "text-embedding-bge-large-en-v1.5",
        "base_url": "http://localhost:1234/v1",
    },
)

result = baseline.request("What inputs were used by the LLM Chat execution?")
print(result["answer"])
print(result["retrieved_objects"])
```
