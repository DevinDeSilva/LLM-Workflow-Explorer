# RExplAnnotator

`RExplAnnotator` is an R package for generating PROV-One RDF annotations for
workflow programs, channels, executions, inputs, outputs, and LLM-assisted
tasks.

```r
library(RExplAnnotator)

prov <- create_prov_module()

program <- prov$provProgram(
  name = "example",
  hasInPort = list(input = list(name = "input", metadata = list())),
  hasOutPort = list(output = list(name = "output", metadata = list()))
)
```
