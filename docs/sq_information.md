# SQ Information

## SQ Library Metadata

To characterize the coverage of the generated `SQ` libraries, we report the schema traversal and filtering statistics for both workflow graphs. This metadata makes the `SQ` generation process more transparent by showing how many ontology-derived paths were explored, how many candidate `SQ`s were produced, and how many remained after graph-specific filtering.

As shown in the table below, both datasets traverse the same ontology schema, covering 14 classes and 385 raw schema paths. However, the final number of retained `SQ`s differs across datasets because the filtering step removes candidate queries that do not return results over the corresponding workflow graph. ChatBS-NexGen retains 252 of 314 generated `SQ`s, while Biomni retains 208 of 270 generated `SQ`s.

### Synthetic Question Generation Metadata

*Path length is measured as the number of class nodes in the schema path.*

| Dataset | Classes | Raw Paths | Raw SQs | Filtered SQs | Avg. Path Length |
|---|---:|---:|---:|---:|---:|
| ChatBS-NexGen | 14 | 385 | 314 | 252 | 4.03 |
| Biomni | 14 | 385 | 270 | 208 | 4.03 |

## SQ vs. Evaluation Questions

The evaluation questions were not used during `SQ` generation, `SQ` selection prompt construction, or retrieval-pipeline development.

Although both `SQ`s and evaluation questions are grounded in the same workflow ontology, the evaluation SPARQL queries were manually authored after `SQ` library construction and were used only to derive ground-truth answers and entity sets:

[Ground-truth evaluation notebook](https://anonymous.4open.science/r/LLM-Workflow-Explorer-040D/evaluations/biomni-base/ground_truth.ipynb)

The `SQ` natural-language questions were not inserted into the evaluation set. Thus, the evaluation measures whether the system can select and compose `SQ`s to answer held-out natural-language questions, rather than whether it can reproduce known `SQ` templates.

## SQ Example

This example retrieves the execution instances responsible for generating data that was used as input to a Large Language Model within the workflow.

### SQ Instance

**ID:**  
`explore_path_workflow:Large_Language_Model_to_provone:Execution`

**Semantic Path:**

```text
workflow:Large_Language_Model
  -> sio:SIO_000230
  -> provone:Data
  -> prov:wasGeneratedBy
  -> provone:Execution
```

**Synthetic Question (`SQ`):**

> What execution(s) generated the data that was used as input in the Large Language Model workflow?

**Input Specification (`I`):**

```json
{"obj": "The URI of the starting object."}
```

**Output Specification (`O`):**

```json
{"value": "The URI of the ending object."}
```

**SPARQL Query Template (`Q_sparql`):**

```sparql
SELECT DISTINCT ?value WHERE {
  <{obj}> sio:SIO_000230 ?a1 .
  ?a1 prov:wasGeneratedBy ?value .
}
```

**Example Instantiated Query (`U_sparql`):**

```sparql
SELECT DISTINCT ?value WHERE {
  <http://testwebsite/testProgram#LLM-id_20260324180131_821>
      sio:SIO_000230 ?a1 .
  ?a1 prov:wasGeneratedBy ?value .
}
```

**Example Output:**

```json
{
  "value": [
    "http://testwebsite/testProgram#id_20260324180129_451"
  ]
}
```

*Example synthetic question instance generated from an ontology path. The instance specifies the semantic path, natural-language question, input/output specification, SPARQL query template, instantiated query, and example output.*
