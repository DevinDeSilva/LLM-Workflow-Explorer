# %%
from pathlib import Path
import asyncio
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Dict, List

import dspy
import pandas as pd
from dotenv import load_dotenv
from IPython.display import display
from pydantic import BaseModel, Field
from tqdm import tqdm
import dycomutils as common_utils
import os
import requests
from bert_score import score as bertscore

from src.config.experiment import FullContextExperimentConfig
from src.experiment.ground_truth import GTInfo, GT
from src.llm import LLM
from src.utils.utils import load_config

REPO_ROOT = Path.cwd()
load_dotenv(REPO_ROOT / ".env")

MetricFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]

class AnswerQualityScores(BaseModel):
    completeness: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    understanderbility: float = Field(ge=0.0, le=1.0)


class AnswerQualityJudgeSignature(dspy.Signature):
    """Score answer completeness, accuracy, and relevance against the ground truth."""

    question: str = dspy.InputField()
    ground_truth_answer: str = dspy.InputField()
    model_answer: str = dspy.InputField()
    completeness: float = dspy.OutputField(desc="How much of the ground truth is covered, from 0 to 1.")
    faithfulness: float = dspy.OutputField(desc="How faithful the answer is to the ground truth, from 0 to 1.")
    relevance: float = dspy.OutputField(desc="How directly the answer addresses the question, from 0 to 1.")
    understanderbility: float = dspy.OutputField(desc="How understandable is the answer by a lay user, from 0 to 1.")


_judge_llm: Any | None = None


def get_judge_llm() -> Any:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = LLM(
            JUDGE_LLM["llm_type"],
            JUDGE_LLM["llm_library"],
            **JUDGE_LLM["llm_config"],
        )
    return _judge_llm


def metric_answer_token_overlap(pred: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    pred_counter = Counter(tokenize(pred.get("answer", "")))
    actual_counter = Counter(tokenize(actual.get("answer", "")))

    overlap = sum((pred_counter & actual_counter).values())
    pred_total = sum(pred_counter.values())
    actual_total = sum(actual_counter.values())

    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / actual_total if actual_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "answer_token_precision": precision,
        "answer_token_recall": recall,
        "answer_token_f1": f1,
    }
    
def metric_bert_score(pred: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    pred_ans = [pred.get("answer", "").replace("\n", " ")]
    actual_ans = [actual.get("answer", "").replace("\n", " ")]
    
    P, R, F1 = bertscore(pred_ans, actual_ans, lang='en', verbose=False)
    return {
        "bertscore_precision": P[0].item(),
        "bertscore_recall": R[0].item(),
        "bertscore_f1": F1[0].item(),
    }
    
    


def metric_ground_truth_entity_coverage(pred: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    gt_values = extract_ground_truth_entity_values(actual)
    pred_values = extract_prediction_surface_forms(pred)
    covered_values = [value for value in gt_values if text_is_covered(value, pred_values)]
    missing_values = [value for value in gt_values if value not in covered_values]

    coverage = len(covered_values) / len(gt_values) if gt_values else float("nan")
    return {
        "gt_entity_total": len(gt_values),
        "gt_entity_covered": len(covered_values),
        "gt_entity_coverage": coverage,
        "missing_gt_entities": json.dumps(missing_values, ensure_ascii=True),
    }


def build_llm_answer_quality_metric() -> MetricFn:
    judge = dspy.Predict(AnswerQualityJudgeSignature)

    def metric(pred: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
        judge_llm = get_judge_llm()
        with dspy.context(lm=judge_llm.llm):
            scores = judge(
                question=actual.get("question", ""),
                ground_truth_answer=actual.get("answer", ""),
                model_answer=strip_citations(pred.get("answer", "")),
            )

        validated_scores = AnswerQualityScores(
            completeness=float(scores.completeness),
            accuracy=float(scores.accuracy),
            relevance=float(scores.relevance),
        )
        return {
            "llm_completeness": validated_scores.completeness,
            "llm_accuracy": validated_scores.accuracy,
            "llm_relevance": validated_scores.relevance,
        }

    metric.__name__ = "llm_answer_quality"
    return metric

def build_nli_metric(pred: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    
    return {
        "nli_score":0
        }

EVALUATION_NAME = "chatbs-base"
CONFIG_FILENAME = "config.fullcontext.yaml"

PREDICTION_FILES = {
    "fullcontext": "evaluations/chatbs-base/explainer/fullcontext/exp__20260420220709/RESULTS.jsonl",
    "llmbased": "evaluations/chatbs-base/explainer/llmbased/exp__20260420235310/RESULTS.jsonl",
    "grasp": "evaluations/chatbs-base/explainer/grasp/exp_202604201325/RESULTS.jsonl",
    "ours":"evaluations/chatbs-base/explainer/results/exp__20260422044127/RESULTS.jsonl"
}

JUDGE_LLM = {
    "llm_type": "openai",
    "llm_library": "dspy",
    "llm_config": {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 300,
    },
}


ENABLED_METRICS: Dict[str, List[MetricFn]] = {
    "multi":[
        metric_answer_token_overlap,
        metric_ground_truth_entity_coverage,
        metric_bert_score,
#        build_llm_answer_quality_metric()
        ], 
    "single--bool":[
        metric_answer_token_overlap,
        metric_ground_truth_entity_coverage,
        metric_bert_score,
#        build_llm_answer_quality_metric()
        ], 
    "single--entity":[
        metric_answer_token_overlap,
        metric_ground_truth_entity_coverage,
        metric_bert_score,
#        build_llm_answer_quality_metric()
        ]

}

MAX_EXAMPLES_PER_RUN = None

SAVE_DIR = REPO_ROOT / "evaluations" / EVALUATION_NAME / "analysis"


# %% [markdown]
# # Evaluation Notebook
# 
# Edit the first cell only for normal use:
# 
# - Set `EVALUATION_NAME` and `CONFIG_FILENAME` to choose the evaluation folder.
# - Add one or more `codename -> RESULTS.jsonl` entries to `PREDICTION_FILES`.
# - Turn on `ENABLE_LLM_JUDGE` if you want the LLM-based grading metric.
# - Add new metric functions in the metric cell and register them in `METRIC_REGISTRY`.
# 

# %%


# %%
def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def strip_citations(text: Any) -> str:
    cleaned = re.sub(
        r"<cite,\s*id=\d+>.*?</cite>",
        " ",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_text(text: Any) -> str:
    text = strip_citations(text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def normalize_question(text: Any) -> str:
    return normalize_text(text)


def unique_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def load_ground_truth_bundle(evaluation_name: str, config_filename: str) -> dict[str, Any]:
    evaluation_dir = resolve_repo_path(Path("evaluations") / evaluation_name)
    config_path = evaluation_dir / config_filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_config = load_config(str(config_path))
    config = FullContextExperimentConfig.model_validate(raw_config)
    gt_path = resolve_repo_path(config.gt.save_loc)
    ground_truth = GTInfo(str(gt_path))
    records = [item.model_dump() for item in ground_truth.gt_info]

    by_id = {record["id"]: record for record in records}
    by_question = {
        normalize_question(record["question"]): record
        for record in records
    }

    return {
        "evaluation_dir": evaluation_dir,
        "config_path": config_path,
        "config": config,
        "ground_truth_path": gt_path,
        "records": records,
        "by_id": by_id,
        "by_question": by_question,
    }


GT_BUNDLE = load_ground_truth_bundle(EVALUATION_NAME, CONFIG_FILENAME)

print(f"Config: {GT_BUNDLE['config_path']}")
print(f"Ground truth: {GT_BUNDLE['ground_truth_path']}")
print(f"Ground-truth examples: {len(GT_BUNDLE["records"])}")

display(
    pd.DataFrame(GT_BUNDLE["records"])[["id", "question", "qtype"]].head()
)


# %%
def grasp_input_config(record:Dict[str, Any]) -> Dict[str, Any]:
    output = record.get("output", {})
    endpoint = output.get("endpoint")
    sparql_query = output.get("sparql")
    evidence: list[dict[str, Any]] = []
    relevant_entities: list[str] = []

    if endpoint and sparql_query:
        try:
            req = requests.post(
                endpoint,
                data={'query':sparql_query},
            )
            req.raise_for_status()
            sparql_result = req.json()

            evidence = sparql_result.get("results", {}).get("bindings", [])
            relevant_entities = unique_preserving_order(
                [
                    binding_value["value"]
                    for row in evidence
                    for binding_value in row.values()
                    if isinstance(binding_value, dict) and binding_value.get("value")
                ]
            )
        except requests.HTTPError as exc:
            error_body = (exc.response.text or "").strip() if exc.response is not None else ""
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            evidence = [{"sparql_error": f"HTTP {status_code}: {error_body or str(exc)}"}]
        except requests.RequestException as exc:
            evidence = [{"sparql_error": str(exc)}]

    question = ""
    for message in record.get("messages", []):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                question = content
                break

    return {
        "_line_number": record["_line_number"],
        "_prediction_path": record["_prediction_path"],
        "answer": output.get("answer", ""),
        "evidence": evidence,
        "id": record.get("id"),
        "question": question,
        "relevant_entities": relevant_entities,
        "time_taken": record.get("elapsed"),
    }


INPUT_AUGMENTATION_MAP = {
    "grasp":grasp_input_config
}


# %%

def read_jsonl(path_value: str | Path) -> list[dict[str, Any]]:
    path = resolve_repo_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    records: list[dict[str, Any]] = []
    
    for line_number, record in enumerate(common_utils.serialization.load_jsonl(path), start=1):
        record["_prediction_path"] = str(path)
        record["_line_number"] = line_number
        records.append(record)

    if len(records)>1:
        print(f"{str(path).split(os.sep)[-3]}\n{sorted(records[0].keys())}\n")
    return records


def load_prediction_runs(
    prediction_files: Mapping[str, str | Path],
    data_augmentations: dict[str, Callable]
    ) -> dict[str, list[dict[str, Any]]]:
    if not prediction_files:
        raise ValueError("Add at least one entry to PREDICTION_FILES in the first cell.")

    runs: dict[str, list[dict[str, Any]]] = {}
    for codename, file_path in prediction_files.items():
        records = read_jsonl(file_path)
        if MAX_EXAMPLES_PER_RUN is not None:
            records = records[:MAX_EXAMPLES_PER_RUN]
            
        runs[codename] = records
        if codename in data_augmentations:
            for i,r in enumerate(runs[codename]):
                runs[codename][i] = data_augmentations[codename](r)
    return runs


PREDICTION_RUNS = load_prediction_runs(
    PREDICTION_FILES,
    INPUT_AUGMENTATION_MAP
    )

prediction_inventory_df = pd.DataFrame(
    [
        {
            "run": codename,
            "file": records[0]["_prediction_path"] if records else "",
            "rows": len(records),
        }
        for codename, records in PREDICTION_RUNS.items()
    ]
)
display(prediction_inventory_df)


# %%
def tokenize(text: Any) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


def extract_ground_truth_entity_values(actual: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for entity in actual.get("entities", []):
        if isinstance(entity, dict):
            for value in entity.values():
                if value is not None and str(value).strip():
                    values.append(str(value))
        elif entity is not None and str(entity).strip():
            values.append(str(entity))
    return unique_preserving_order(values)


def extract_prediction_surface_forms(pred: dict[str, Any]) -> list[str]:
    values: list[str] = []

    answer = strip_citations(pred.get("answer", ""))
    if answer:
        values.append(answer)

    for entity in pred.get("relevant_entities", []):
        if isinstance(entity, dict):
            for key in ("id", "label"):
                value = entity.get(key)
                if value is not None and str(value).strip():
                    values.append(str(value))
            for value in entity.get("types", []) or []:
                if value is not None and str(value).strip():
                    values.append(str(value))
        elif entity is not None and str(entity).strip():
            values.append(str(entity))

    for triple in pred.get("evidence", []):
        if isinstance(triple, dict):
            for key in (
                "subject_id",
                "subject_label",
                "predicate_id",
                "predicate_label",
                "object_id",
                "object_label",
            ):
                value = triple.get(key)
                if value is not None and str(value).strip():
                    values.append(str(value))
        elif triple is not None and str(triple).strip():
            values.append(str(triple))

    return unique_preserving_order(values)


def text_is_covered(target: str, candidates: list[str]) -> bool:
    target_tokens = set(tokenize(target))
    if not target_tokens:
        return False

    for candidate in candidates:
        candidate_tokens = set(tokenize(candidate))
        if not candidate_tokens:
            continue
        if target_tokens <= candidate_tokens:
            return True
    return False





# %%

# %%
def resolve_ground_truth_record(pred: dict[str, Any]) -> Dict[str, Any]:
    pred_id = pred.get("id")
    if pred_id in GT_BUNDLE["by_id"]:
        return GT_BUNDLE["by_id"][pred_id]

    question_key = normalize_question(pred.get("question", ""))
    gt = GT_BUNDLE["by_question"].get(question_key)
    
    if not gt:
        ValueError("No GT found.")
        
    return gt


def evaluate_run(
    run_name: str,
    predictions: list[dict[str, Any]],
    metric_functions: Dict[str,List[MetricFn]],
) -> dict[str, List[dict[str, Any]]]:
    rows_cat: dict[str, List[dict[str, Any]]] = {}

    for pred in tqdm(predictions):
        actual = resolve_ground_truth_record(pred)
        row: dict[str, Any] = {
            "run": run_name,
            "prediction_path": pred.get("_prediction_path"),
            "prediction_line": pred.get("_line_number"),
            "prediction_id": pred.get("id"),
            "ground_truth_found": actual is not None,
            "question": pred.get("question", ""),
            "pred_answer": strip_citations(pred.get("answer", "")),
            "prediction_time_taken": pred.get("time_taken"),
        }
        
        for atags, func in metric_functions.items():
            satags = atags.split("--")
            if not actual["qtype"]:
                ValueError(f"tags is None for quession :{pred.get("id")}")
            if len(set(satags) - set(actual["qtype"]))>0:
                continue

            row.update(
                {
                    "ground_truth_id": actual.get("id"),
                    "ground_truth_qtype": json.dumps(actual.get("qtype", []), ensure_ascii=True),
                    "ground_truth_tags": json.dumps(actual.get("tags", {}), ensure_ascii=True, sort_keys=True),
                    "ground_truth_answer": actual.get("answer", ""),
                }
            )

            for metric_fn in func:
                try:
                    row.update(metric_fn(pred, actual))
                except Exception as exc:
                    row[f"{metric_fn.__name__}_error"] = str(exc)

            if atags not in rows_cat:
                rows_cat[atags] = []
            rows_cat[atags].append(row)

    return rows_cat


evaluation_rows: dict[str, List[dict[str, Any]]] = {}
for run_name, predictions in PREDICTION_RUNS.items():
    _temp =  evaluate_run(run_name, predictions, ENABLED_METRICS)
    for k,v in _temp.items():
        if k not in evaluation_rows:
            evaluation_rows[k] = []
        
        evaluation_rows[k].extend(v)


# %%
def build_run_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    counts_df = (
        results_df.groupby("run", dropna=False)
        .agg(
            evaluated_examples=("run", "size"),
            matched_ground_truth=("ground_truth_found", "sum"),
            avg_prediction_time_taken=("prediction_time_taken", "mean"),
        )
        .reset_index()
    )

    numeric_columns = [
        column
        for column in results_df.select_dtypes(include="number").columns
        if column not in {"prediction_line", "prediction_time_taken"}
    ]

    if not numeric_columns:
        return counts_df

    summary_df = (
        results_df.groupby("run", dropna=False)[numeric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "run" if column == ("run", "") else f"{column[0]}_{column[1]}"
        for column in summary_df.columns
    ]

    return counts_df.merge(summary_df, on="run", how="left")

for cat, eval_rows in evaluation_rows.items():
    print(cat)
    RESULTS_DF = pd.DataFrame(eval_rows)
    if not RESULTS_DF.empty:
        RESULTS_DF = RESULTS_DF.sort_values(
            ["run", "ground_truth_found", "prediction_line"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    display(RESULTS_DF.head())
    RUN_SUMMARY_DF = build_run_summary(RESULTS_DF)
    display(RUN_SUMMARY_DF)


    os.makedirs(os.path.join(SAVE_DIR, cat), exist_ok=True)
    RESULTS_DF.to_csv(os.path.join(SAVE_DIR, cat, "results.csv"), index=False)
    RUN_SUMMARY_DF.to_csv(os.path.join(SAVE_DIR, cat, "results_summary.csv"), index=False)
