import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("dataset", ["chatbs-base", "biomni-base"])
@pytest.mark.parametrize("gemma_has_scores", [False, True])
def test_paper_ensemble_counts_only_valid_judges_and_pools_valid_items(dataset, gemma_has_scores):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "evaluations" / dataset / "paper_summary_tables.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    setup = ast.parse(cells["cdc0369c"])
    judge_assignment = next(node for node in setup.body if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == "JUDGE_IDS" for target in node.targets))
    judges = ast.literal_eval(judge_assignment.value)
    assert judges == ["gpt-5.4-mini", "qwen3.6-35b-a3b", "gemma-4-31b"]
    assert notebook["cells"][-1]["id"] == "krippendorff-alpha-results"
    definitions = ast.parse(cells["294bec8c"])
    selected = [node for node in definitions.body if
                (isinstance(node, ast.FunctionDef) and node.name == "aggregate_judge_values") or
                (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "LLM_METRICS" for target in node.targets))]
    namespace = {"pd": pd, "np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "aggregation", "exec"), namespace)
    frames = {}
    for judge_index, judge in enumerate(judges):
        for category_index, category in enumerate(("entity", "bool", "numeric")):
            value = round(0.2 * (1 + judge_index + category_index), 2)
            row = {"run": "ours", "ground_truth_id": category,
                   "answer_token_f1": 0.5, "bertscore_f1": 0.5, "nli_entailment_max": 0.5,
                   "entity_recall_final": value if category == "entity" else 1.0,
                   "entity_recall_total": value if category == "entity" else 1.0}
            if judge != "gemma-4-31b" or gemma_has_scores:
                row.update({metric: value for metric in namespace["LLM_METRICS"]})
            rows = [row]
            if category == "entity":
                rows.append({**row, "ground_truth_id": "missing",
                             **{metric: np.nan for metric in namespace["LLM_METRICS"] if metric in row},
                             "entity_recall_final": np.nan, "entity_recall_total": np.nan})
            frames[Path(judge) / category / "results.csv"] = pd.DataFrame(rows)
    namespace.update(JUDGE_IDS=judges, CATEGORY_NAMES=["bool", "entity", "numeric"],
                     judge_analysis_dir=lambda judge: Path(judge),
                     read_csv_or_empty=lambda path: frames[path].copy(),
                     display=lambda *args: None, round_digits=3)
    exec(cells["b2c64774"], namespace)
    exec(cells["7da7d60a"], namespace)
    row = namespace["overall_summary_table"].iloc[0]
    expected_judges = 3 if gemma_has_scores else 2
    assert row["Result Sources"] == 3
    assert row["LLM Judges"] == expected_judges
    assert row["N"] == row["N Min"] == row["N Max"] == 4
    assert row["Recall N"] == 1
    assert row["Entity Recall"] == row["Entity Recall Total"] == 40
    assert row["Entity Recall SD"] == row["Entity Recall Total SD"] == 20
    for attribute in ("Completeness", "Relevance", "Faithfulness", "Understandability"):
        assert row[f"{attribute} Judges"] == expected_judges
        assert row[attribute] == (60 if gemma_has_scores else 50)
        assert row[f"{attribute} SD"] == pytest.approx(20 if gemma_has_scores else 14.142, abs=0.00051)
    source = namespace["overall_judge_source"]
    gemma = source.loc[source.judge_id.eq("gemma-4-31b")].iloc[0]
    assert gemma["llm_completeness_count"] == (3 if gemma_has_scores else 0)
