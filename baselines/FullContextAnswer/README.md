# FullContextAnswer Baseline

`FullContextAnswerBaseline` is an LLM-backed baseline for workflow explanation over a provenance knowledge graph.

## Goal

This baseline answers a question by sending the model:

- the application description
- namespace prefixes
- a compact schema summary
- the full execution KG
- the user question

It mirrors the `LLMbased` request interface and returns a dictionary with:

- `answer`: the model's grounded answer
- `relevant_entities`: entities the model cited or that were recovered from the answer
- `evidence`: supporting triples the model cited when available

## Usage

```python
from baselines.FullContextAnswer import FullContextAnswerBaseline

baseline = FullContextAnswerBaseline(
    kg_path="usecases/chatbs/data/1_sample_graph/chatbs_sample.ttl",
    ontology_path="schema/WorkFlow.ttl",
    schema_json_path="schema/schemaV2.json",
    metadata_path="usecases/chatbs/data/1_sample_graph/chatbs_sample_metadata.json",
    application_description="A program to analyze and generate provenance data for ChatBS-NexGen.",
    llm_config={"model": "gpt-4o-mini"},
)

result = baseline.request("What happened to the generated answer?")
print(result["answer"])
print(result["relevant_entities"])
print(result["evidence"])
```

## Method

Unlike `LLMbased`, this baseline does not perform local grounding-first reasoning. Instead it packages the whole graph context directly into a single inference request and asks the LLM to return:

1. a concise answer
2. the key referenced entities
3. supporting evidence triples

The implementation resolves the model's returned entity and predicate references back onto the KG so the response format stays aligned with the repository's other explanation baseline.
