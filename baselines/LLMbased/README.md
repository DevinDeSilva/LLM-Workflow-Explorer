# LLMbased Baseline

`GroundedWorkflowBaseline` is a lightweight baseline for workflow explanation over a provenance knowledge graph.

## Goal

The baseline answers a natural-language question by:

1. grounding the question to entities in the KG
2. collecting nearby provenance evidence
3. synthesizing a short natural-language explanation
4. returning both the explanation and the KG entities used

It loads:

- an execution KG in Turtle
- the workflow ontology Turtle file
- the schema summary JSON
- optional KG metadata with namespace prefixes

and exposes a single request-style interface:

```python
from baselines.LLMbased import GroundedWorkflowBaseline

baseline = GroundedWorkflowBaseline(
    kg_path="usecases/chatbs/data/1_sample_graph/chatbs_sample.ttl",
    ontology_path="schema/WorkFlow.ttl",
    schema_json_path="schema/schemaV2.json",
    metadata_path="usecases/chatbs/data/1_sample_graph/chatbs_sample_metadata.json",
)

result = baseline.request("What happened to the generated answer?")
print(result["answer"])
print(result["relevant_entities"])
print(result["evidence"])
```

The returned object is a dictionary with:

- `answer`: grounded natural-language explanation
- `relevant_entities`: KG entities used to ground the explanation
- `evidence`: the highest-scoring supporting triples from the local graph neighborhood

## Method

The baseline is fully local and deterministic. It does not call an LLM. The method is:

### 1. Load the workflow context

The system loads four sources:

- the execution KG
- the ontology Turtle file
- the schema summary JSON
- optional metadata for namespace prefixes

The ontology and schema are used to interpret the meaning of workflow classes and relations such as:

- `provone:Execution`
- `provone:Program`
- `provone:Data`
- `workflow:Large_Language_Models`
- `prov:used`
- `prov:wasGeneratedBy`
- `prov:qualifiedUsage`
- `prov:qualifiedGeneration`

### 2. Build an entity index over the KG

Every URI node in the KG is indexed. For each entity the baseline stores:

- URI and CURIE form
- RDF types
- labels
- descriptions
- identifiers
- literal values and literal facts

If an entity has no readable label, the baseline derives one from the URI fragment. For example, a URI like `...#Data-id_...-generated_answer` becomes a readable label such as `generated answer`.

This creates a searchable representation for each node without requiring SPARQL program generation.

### 3. Detect the question intent

The user query is tokenized and mapped to one or more coarse intents using keyword rules. The current intents are:

- `model`
- `program`
- `execution`
- `input`
- `output`
- `agent`
- `time`
- `explanation`

These intents are used to bias grounding toward the right kinds of entities. For example:

- model questions favor `workflow:Large_Language_Models`
- execution questions favor `provone:Execution`
- input and output questions favor `provone:Data`, `provone:Collection`, and `provone:Execution`

### 4. Ground the query to KG entities

Each indexed entity is scored against the user query using a heuristic ranking function.

The score combines:

- token overlap between the query and the entity text
- label overlap
- exact substring matches on labels or CURIEs
- matches against literal values
- schema/type bonuses from the detected intent

Examples:

- if the query mentions `LLM Chat`, executions tied to the `LLM Chat` program get extra weight
- if the query asks about a `model`, LLM/model nodes get extra weight
- if the query asks about a `prompt`, nodes connected to prompt inputs are boosted

The top-ranked entities become the anchors for evidence collection.

### 5. Expand a local evidence neighborhood

Starting from the grounded anchor entities, the baseline walks outward through the KG for a small number of hops.

It inspects both:

- outgoing triples from the entity
- incoming triples into the entity

Each candidate triple is scored using:

- overlap with the query terms
- whether the predicate is a workflow-relevant predicate
- whether the triple touches an anchored entity
- intent-specific boosts for predicates such as `prov:used`, `prov:wasGeneratedBy`, `prov:wasAssociatedWith`, or `workflow:llm_model`

This produces a ranked set of evidence triples and a ranked set of nearby relevant entities.

### 6. Choose a focus entity

The answer is not built from every matched entity equally. The baseline picks a focus entity based on the detected intent.

Examples:

- model questions prefer LLM/model nodes
- input questions prefer executions
- output questions prefer data entities
- program questions prefer program nodes

This keeps the answer centered on the part of the workflow the question is most likely about.

### 7. Generate a typed explanation

The baseline has hand-written explanation templates for major workflow entity types:

- execution
- program
- LLM/model node
- data or collection node
- agent node

Each template pulls structured provenance facts from the graph. For example:

- executions summarize program, model, user, time, inputs, and outputs
- data nodes summarize value preview, producer execution, and downstream usage
- model nodes summarize model name, consumed inputs, produced outputs, and linked executions

If no typed template fits well, the system falls back to a short statement built directly from the top evidence triples.

### 8. Return grounded output

The result has three parts:

- `answer`: a concise natural-language explanation
- `relevant_entities`: the highest-priority KG entities used for grounding
- `evidence`: the best supporting triples from the entity neighborhood

This makes the baseline easy to compare against more advanced approaches because the reasoning path is inspectable.

## Why This Is a Baseline

This method is intentionally simple:

- no external retrieval system
- no vector database
- no LLM generation
- no dynamic SPARQL planning

Instead, it uses direct graph inspection plus schema-aware heuristics. That makes it useful as a comparison point for richer explainer systems in the repository.

## Current Limitations

The method is interpretable, but it is also constrained:

- grounding depends on lexical overlap and simple intent rules
- it only explores a bounded local neighborhood
- it does not decompose complex multi-part questions
- it does not learn from examples
- answers are template-based rather than fully generative

So the baseline is strongest on concrete provenance questions such as:

- which model handled this step
- what inputs were used
- what generated this output
- which program ran
- who was associated with an execution
