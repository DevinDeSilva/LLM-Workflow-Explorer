## Latency and Call Count Analysis

The table below reports the average latency and call counts for each method across the ChatBS-NexGen and Biomni benchmarks. Vector-based and graph-RAG baselines require fewer LLM calls and therefore achieve lower latency, while GRASP incurs additional cost from iterative SPARQL generation and execution. Our method has the highest latency, particularly on Biomni, because it performs multiple LLM-guided steps for SQ selection, parameter grounding, answer validation, and refinement.

This reflects a trade-off between efficiency and explanation quality: as shown in the main evaluation results, the additional calls enable stronger grounding and higher-quality explanations, but reduce suitability for latency-sensitive settings.

### Dataset-Specific Latency and Call Counts

*Average latency is reported in seconds. SPARQL calls denote explicit query-bearing tool calls for GRASP. Dashes indicate metrics not available for the corresponding method.*

| Method | ChatBS-NexGen Latency (s) | ChatBS-NexGen LLM Calls | ChatBS-NexGen SPARQL Calls | Biomni Latency (s) | Biomni LLM Calls | Biomni SPARQL Calls |
|---|---:|---:|---:|---:|---:|---:|
| VSB | 2.94 | 1.00 | - | 3.18 | 1.00 | - |
| GRASP | 15.49 | 7.79 | 5.79 | 10.57 | 5.06 | 1.63 |
| HippoRAG | 4.59 | 1.00 | - | 4.83 | 1.00 | - |
| HyperGRAG | 12.67 | 1.74 | - | 14.11 | 1.91 | - |
| Ours | 39.98 | 6.95 | 2.59 | 146.06 | 9.32 | 2.74 |

## Developer Workload

The table below summarizes the effort needed to add provenance capture to the two systems. Both systems required a small number of high-level annotation calls. ChatBS-NexGen required wrapping a fixed pipeline, while Biomni required more adapter code to capture dynamic agent iterations, code execution, critic steps, and intermediate artifacts. Instrumentation time was not recorded.

### Instrumentation Overhead

*Files denotes modified files, Calls denotes annotation calls, LOC denotes added lines of code, Time denotes instrumentation time, and Triples denotes produced RDF triples. `-` indicates not recorded.*

| System | Files | Calls | LOC | Time | Triples |
|---|---:|---:|---:|---:|---:|
| ChatBS-NexGen | 5 | 30 | 870 | - | 895 |
| Biomni | 3 | 31 | 1,172 | - | 2,022 |
