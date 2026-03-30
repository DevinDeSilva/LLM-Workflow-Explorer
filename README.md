# LLM-Workflow-Explorer

## Query dependency planning

For QA over a knowledge graph, converting a user question into an extraction dependency graph is a sound intermediate step.
It forces the system to separate:

- the answer shape it needs to produce,
- the entities and constraints it must resolve first,
- the graph facts or traversals that depend on those earlier resolutions.

This repository now includes a basic DSPy-backed planner in `src/planning/dependancy_graph.py`.
The planner asks an LLM for a structured requirement graph, then normalizes the result into:

- `requirements`: nodes representing extractable facts, entity resolution steps, or graph traversals,
- `edges`: dependencies between those nodes,
- `execution_order`: a linearized order that can drive downstream retrieval.

The implementation is intentionally minimal.
If the DSPy output is malformed, it falls back to a deterministic two-step plan so the research pipeline remains debuggable while you iterate on prompts and schema grounding.
