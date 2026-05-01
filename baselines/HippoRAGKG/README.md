# HippoRAG KG Adapter

This adapter runs the vendored `baselines/HippoRAG` code over our ChatBS KG
without patching HippoRAG internals.

The low-friction route is to inject KG-derived OpenIE records:

1. Convert the TTL into one document per URI subject.
2. Write HippoRAG-compatible OpenIE JSON where `extracted_entities` and
   `extracted_triples` come directly from RDF triples.
3. Let HippoRAG run its normal indexing, retrieval, QA, and write `RESULTS.jsonl`
   in the same shape as the other baselines.

Prepare the corpus and injected OpenIE file without importing HippoRAG:

```sh
uv run python baseline_HippoRAG.py --evaluation chatbs-base --prepare-only
```

Run a small smoke evaluation:

```sh
uv run python baseline_HippoRAG.py --evaluation chatbs-base --limit-questions 3
```

Run the full evaluation:

```sh
uv run python baseline_HippoRAG.py --evaluation chatbs-base
```

The default config is `evaluations/chatbs-base/config.hipporag.yaml`. It uses
LM Studio's OpenAI-compatible API for both chat and embeddings. Set
`inject_kg_openie: "false"` or pass `--no-kg-openie-injection` if you want
HippoRAG to run its own LLM-based OpenIE over the object documents.

The full HippoRAG run still needs HippoRAG's own dependencies, including
`python_igraph`. If the repo's default environment does not have them, run the
same command from a HippoRAG-compatible environment after installing
`baselines/HippoRAG/requirements.txt`.
