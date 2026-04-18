# LLMbased Baseline

`GroundedWorkflowBaseline` is a grounding-first workflow explanation baseline.

It initializes from:

- an execution KG in Turtle
- the workflow ontology in `schema/WorkFlow.ttl`
- the filtered schema in `schema/schemaV2.json`

It exposes a `request(user_query)` method that returns:

- `answer`: a natural-language answer grounded in the KG
- `relevant_entities`: the main KG entities used to answer
- `evidence`: the supporting triples used for the answer

The baseline works deterministically without an external model. If you supply an LLM or `llm_config`, it uses `src.llm` only to rewrite the already-grounded answer more naturally.
