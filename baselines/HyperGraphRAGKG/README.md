# HyperGraphRAG KG Adapter

This adapter runs the vendored `baselines/HyperGraphRAG` code over our ChatBS
KG without modifying the submodule.

The low-friction route is to convert RDF triples into HyperGraphRAG's runtime
shape:

1. Convert the TTL into one chunk per URI subject.
2. Represent each RDF triple as a hyperedge node.
3. Connect each hyperedge to subject/object entity nodes.
4. Populate HyperGraphRAG's KV, NetworkX, entity vector, hyperedge vector, and
   chunk vector stores.

Prepare the custom KG JSON without importing HyperGraphRAG:

```sh
uv run python baseline_HyperGraphRAG.py --evaluation chatbs-base --prepare-only
```

Run a small smoke evaluation:

```sh
uv run python baseline_HyperGraphRAG.py --evaluation chatbs-base --limit-questions 3
```

Run the full evaluation:

```sh
uv run python baseline_HyperGraphRAG.py --evaluation chatbs-base
```

The default config is `evaluations/chatbs-base/config.hypergraphrag.yaml`. It
uses LM Studio's OpenAI-compatible API for chat and embeddings.

The full run needs HyperGraphRAG's own dependencies, especially
`nano-vectordb`, `networkx`, and `graspologic`. If the repo's default
environment does not have them, run the command from an environment with
`baselines/HyperGraphRAG/requirements.txt` installed.

