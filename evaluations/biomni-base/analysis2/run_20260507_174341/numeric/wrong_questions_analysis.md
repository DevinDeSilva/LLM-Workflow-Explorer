# Numeric Wrong Questions Analysis

Source: results.csv
Total evaluated rows: 147 (7 runs x 21 questions)
Rows marked wrong by numeric_accuracy: 124

## Important scoring caveat

The evaluator uses extract_numeric_decision() from biomni_analysis2.py. It first looks for digits near words such as "is", "count", or "total"; if that fails, it returns the first standalone integer in the answer. This creates false negatives when an answer says "exactly one" in words but mentions "Step 4" or an ID before a digit 1. I labeled these as likely evaluator false negatives.

## Accuracy by run

| run | correct | wrong | observed accuracy | likely scoring false negatives | heuristic adjusted correct | heuristic adjusted accuracy |
|---|---:|---:|---:|---:|---:|---:|
| fullcontext | 0 | 21 | 0.0% | 0 | 0 | 0.0% |
| grasp | 4 | 17 | 19.0% | 1 | 5 | 23.8% |
| hipporag | 3 | 18 | 14.3% | 0 | 3 | 14.3% |
| hypergraphrag | 2 | 19 | 9.5% | 6 | 8 | 38.1% |
| llmbased | 3 | 18 | 14.3% | 0 | 3 | 14.3% |
| ours | 11 | 10 | 52.4% | 6 | 17 | 81.0% |
| vectorsimilarity | 0 | 21 | 0.0% | 0 | 0 | 0.0% |

## Failure reason breakdown

| reason label | rows |
|---|---:|
| no answer / context missing | 38 |
| returned/extracted the step number instead of the count | 24 |
| likely evaluator false negative: answer says one, extractor grabbed another number | 13 |
| returned identifier suffix instead of count | 10 |
| retrieved or described the wrong workflow entity | 9 |
| counted both outputs/generations instead of the requested output/port | 8 |
| wrong total experiment count | 7 |
| wrong numeric value | 7 |
| tool/SPARQL wrapper confused numeric extraction | 4 |
| no numeric count extracted | 4 |

## Failure reasons by run

| run | top reasons |
|---|---|
| fullcontext | no answer / context missing (20); wrong total experiment count (1) |
| grasp | no answer / context missing (12); tool/SPARQL wrapper confused numeric extraction (3); wrong total experiment count (1); likely evaluator false negative: answer says one, extractor grabbed another number (1) |
| hipporag | returned/extracted the step number instead of the count (8); returned identifier suffix instead of count (4); no numeric count extracted (2); counted both outputs/generations instead of the requested output/port (2); wrong total experiment count (1); wrong numeric value (1) |
| hypergraphrag | returned/extracted the step number instead of the count (7); likely evaluator false negative: answer says one, extractor grabbed another number (6); counted both outputs/generations instead of the requested output/port (2); no answer / context missing (2); wrong total experiment count (1); returned identifier suffix instead of count (1) |
| llmbased | retrieved or described the wrong workflow entity (9); wrong numeric value (5); counted both outputs/generations instead of the requested output/port (2); wrong total experiment count (1); returned/extracted the step number instead of the count (1) |
| ours | likely evaluator false negative: answer says one, extractor grabbed another number (6); counted both outputs/generations instead of the requested output/port (2); wrong total experiment count (1); returned identifier suffix instead of count (1) |
| vectorsimilarity | returned/extracted the step number instead of the count (8); no answer / context missing (4); returned identifier suffix instead of count (4); no numeric count extracted (2); wrong total experiment count (1); wrong numeric value (1); tool/SPARQL wrapper confused numeric extraction (1) |

## Per-question summary

| GT id | expected | correct runs | wrong runs | dominant wrong reason | question |
|---|---:|---:|---:|---|---|
| gt_0 | 12 | 0 | 7 | wrong total experiment count | How many "experiment execution" are there in this? |
| gt_86 | 1 | 1 | 6 | no answer / context missing | what is the number of output for the "Step 10 - next agent step output" in the execution "http://testwebsite/testProgram#id_20260505001035_720" ? |
| gt_87 | 1 | 1 | 6 | no answer / context missing | what is the number of output for the "Step 10 - next_step output port for critic review 1" in the execution "http://testwebsite/testProgram#id_2026... |
| gt_88 | 1 | 4 | 3 | no answer / context missing | what is the number of output for the "Step 10 - critic feedback output" in the execution "http://testwebsite/testProgram#id_20260505001035_720" ? |
| gt_89 | 1 | 1 | 6 | no answer / context missing | what is the number of output for the "Step 10 - critic_feedback output port for critic review 1" in the execution "http://testwebsite/testProgram#i... |
| gt_90 | 1 | 1 | 6 | no answer / context missing | what is the number of output for the "Step 4 - next agent step output" in the execution "http://testwebsite/testProgram#id_20260505001006_475" ? |
| gt_91 | 1 | 1 | 6 | returned/extracted the step number instead of the count | what is the number of output for the "Step 4 - next_step output port for generation iteration 1" in the execution "http://testwebsite/testProgram#i... |
| gt_92 | 1 | 0 | 7 | no answer / context missing | what is the number of output for the "Step 4 - LLM response output" in the execution "http://testwebsite/testProgram#id_20260505001006_475" ? |
| gt_93 | 1 | 4 | 3 | no answer / context missing | what is the number of output for the "Step 4 - response output port for generation iteration 1" in the execution "http://testwebsite/testProgram#id... |
| gt_94 | 1 | 1 | 6 | returned/extracted the step number instead of the count | what is the number of output for the "Step 5 - next agent step output" in the execution "http://testwebsite/testProgram#id_20260505001009_972" ? |
| gt_95 | 1 | 0 | 7 | no answer / context missing | what is the number of output for the "Step 5 - next_step output port for generation iteration 2" in the execution "http://testwebsite/testProgram#i... |
| gt_96 | 1 | 2 | 5 | no answer / context missing | what is the number of output for the "Step 5 - LLM response output" in the execution "http://testwebsite/testProgram#id_20260505001009_972" ? |
| gt_97 | 1 | 0 | 7 | counted both outputs/generations instead of the requested output/port | what is the number of output for the "Step 5 - response output port for generation iteration 2" in the execution "http://testwebsite/testProgram#id... |
| gt_98 | 1 | 1 | 6 | no answer / context missing | what is the number of output for the "Step 6 - next agent step output" in the execution "http://testwebsite/testProgram#id_20260505001011_890" ? |
| gt_99 | 1 | 0 | 7 | returned/extracted the step number instead of the count | what is the number of output for the "Step 6 - next_step output port for generation iteration 3" in the execution "http://testwebsite/testProgram#i... |
| gt_100 | 1 | 1 | 6 | no answer / context missing | what is the number of output for the "Step 6 - LLM response output" in the execution "http://testwebsite/testProgram#id_20260505001011_890" ? |
| gt_101 | 1 | 0 | 7 | no answer / context missing | what is the number of output for the "Step 6 - response output port for generation iteration 3" in the execution "http://testwebsite/testProgram#id... |
| gt_102 | 1 | 1 | 6 | no answer / context missing | what is the number of output for the "Step 8 - next agent step output" in the execution "http://testwebsite/testProgram#id_20260505001016_956" ? |
| gt_103 | 1 | 2 | 5 | no answer / context missing | what is the number of output for the "Step 8 - next_step output port for generation iteration 4" in the execution "http://testwebsite/testProgram#i... |
| gt_104 | 1 | 1 | 6 | returned identifier suffix instead of count | what is the number of output for the "Step 8 - LLM response output" in the execution "http://testwebsite/testProgram#id_20260505001016_956" ? |
| gt_105 | 1 | 1 | 6 | no answer / context missing | what is the number of output for the "Step 8 - response output port for generation iteration 4" in the execution "http://testwebsite/testProgram#id... |

## Main findings

- gt_0 is the only global count question and every run misses it. The ground truth is 12 experiment executions, but methods returned blank, 0, 2, 7, or 16.
- Most step/output questions have ground truth count 1. A large share of failures are not counting failures in the graph; they are answer-format or extraction failures where the model mentions a step number before the actual count.
- fullcontext mainly says the requested execution is missing from its provided graph context. That accounts for 20 of its 21 failures.
- ours has 10 observed failures, but 6 look like scoring false negatives: the answer says the requested output count is one, while the extractor records Step 4, Step 5, Step 6, Step 8, or an ID-related number. Its likely genuine misses are gt_0, gt_90, gt_94, and gt_99.
- hypergraphrag has a similar issue: 6 of 19 observed failures look like answers that say one but were scored against a preceding step number.
- hipporag and vectorsimilarity often return a bare step number or ID suffix, such as 90, 976, 622, or 949, instead of the count.
- llmbased frequently retrieves/describes a related workflow step rather than answering the requested count.

## Recommended next checks

- Fix or supplement extract_numeric_decision() so it recognizes number words like "one", parses structured JSON answer fields before wrapper text, and prioritizes final/count phrases over step labels and IDs.
- Consider prompting every method to end numeric answers with a machine-readable suffix such as "Final count: <integer>".
- Re-run the numeric evaluation after extractor changes; the observed ranking will likely change, especially for ours and hypergraphrag.
- For genuine model errors, inspect gt_0 plus the rows labeled counted both outputs/generations, returned identifier suffix, and no answer/context missing.
