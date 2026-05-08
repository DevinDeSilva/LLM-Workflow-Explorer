#!/usr/bin/env python
"""Run Biomni evaluations without LLM/NLI metrics and print one overall table.

This is intentionally separate from evaluation_results.py because that file has
top-level execution side effects and writes to the configured analysis folder.
The default output here is evaluations/biomni-base/analysis2.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import dycomutils as common_utils
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.experiment import FullContextExperimentConfig
from src.evaluation.report_builders import augment_ours_record_with_reports
from src.experiment.ground_truth import GTInfo
from src.synthetic_questions import SQRetriver
from src.utils.utils import load_config


MetricFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]

REMOVED_METRICS = {"bert_score", "llm_answer_quality", "nli_entailment"}
FORCED_METRICS = {"entity_retrieval"}

METHOD_LABELS = {
    "fullcontext": "FCB",
    "grasp": "GRASP",
    "hipporag": "HippoRAG",
    "hypergraphrag": "HyperGRAG",
    "llmbased": "GWB",
    "LWE": "Ours",
    "ours": "Ours",
    "vectorsimilarity": "VSB",
}
METHOD_ORDER = ["FCB", "GWB", "VSB", "GRASP", "HippoRAG", "HyperGRAG", "Ours"]

OVERALL_METRIC_COLUMNS = [
    "answer_token_f1",
    "gt_entity_coverage",
    "entity_recall_final",
    "entity_precision_final",
    "entity_f1_final",
    "entity_recall_total",
    "entity_precision_total",
    "entity_f1_total",
]

COLUMN_LABELS = {
    "run": "Method",
    "evaluated_examples": "N",
    "matched_ground_truth": "Matched GT",
    "answer_token_f1": "Answer Token F1",
    "gt_entity_coverage": "GT Entity Coverage",
    "entity_recall_final": "Entity Recall Final",
    "entity_precision_final": "Entity Precision Final",
    "entity_f1_final": "Entity F1 Final",
    "entity_recall_total": "Entity Recall Total",
    "entity_precision_total": "Entity Precision Total",
    "entity_f1_total": "Entity F1 Total",
}

COUNT_COLUMNS = {"N", "Matched GT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Biomni methods without BERTScore, LLM, or NLI metrics "
            "and print only the overall summary table."
        )
    )
    parser.add_argument(
        "--evaluation",
        default="biomni-base",
        help="Folder under evaluations/ containing config.evaluation.yaml.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help=(
            "Output directory. Defaults to evaluations/<evaluation>/analysis2. "
            "If it already contains files, a timestamped subfolder is used."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional method codenames to evaluate. Defaults to all configured methods.",
    )
    parser.add_argument(
        "--max-examples-per-run",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--answer-report",
        default=None,
        help="Override metrics.answer_report for the Ours augmentation.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run and display the table without writing CSV outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Write directly into --save-dir even when it already contains files.",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=3,
        help="Digits to round in the displayed/saved overall table.",
    )
    return parser.parse_args()


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
        if candidate_tokens and target_tokens <= candidate_tokens:
            return True
    return False


def clean_entity_value(value: Any) -> str:
    text = strip_citations(value)
    text = re.sub(r"@[a-zA-Z-]+(?:\^\^<[^>]+>)?$", "", text)
    text = re.sub(r"\^\^<[^>]+>$", "", text)
    return text.strip().strip("<>").strip()


def entity_aliases(value: Any) -> set[str]:
    text = clean_entity_value(value)
    if not text:
        return set()

    aliases = {normalize_text(text)}
    compact_texts = {text}
    if "http://testwebsite/testProgram#" in text:
        compact_texts.update(
            text.replace("http://testwebsite/testProgram#", prefix)
            for prefix in ("Biomni:", "ChatBS-NexGen:")
        )

    for compact_text in compact_texts:
        aliases.add(normalize_text(compact_text))

        for delimiter in ("#", "/", ":"):
            if delimiter in compact_text:
                aliases.add(normalize_text(compact_text.rsplit(delimiter, 1)[-1]))

    return {alias for alias in aliases if alias}


def entity_key(value: Any) -> str:
    aliases = entity_aliases(value)
    if not aliases:
        return ""
    return sorted(aliases, key=lambda alias: (len(alias), alias))[0]


def entity_key_set(values: list[str]) -> set[str]:
    return {key for key in (entity_key(value) for value in values) if key}


def entities_match(left: Any, right: Any) -> bool:
    return bool(entity_aliases(left) & entity_aliases(right))


def unique_entity_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        key = entity_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        unique_values.append(clean_entity_value(value))
    return unique_values


def retrieval_scores(gt_values: list[str], retrieved_values: list[str]) -> dict[str, float]:
    gt_unique = unique_entity_values(gt_values)
    retrieved_unique = unique_entity_values(retrieved_values)

    gt_matched = [
        gt_value
        for gt_value in gt_unique
        if any(entities_match(gt_value, retrieved_value) for retrieved_value in retrieved_unique)
    ]
    retrieved_matched = [
        retrieved_value
        for retrieved_value in retrieved_unique
        if any(entities_match(retrieved_value, gt_value) for gt_value in gt_unique)
    ]

    recall = len(gt_matched) / len(gt_unique) if gt_unique else float("nan")
    precision = (
        len(retrieved_matched) / len(retrieved_unique)
        if retrieved_unique
        else float("nan")
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall)
        else float("nan")
    )
    return {"recall": recall, "precision": precision, "f1": f1}


def collect_binding_values(value: Any) -> list[str]:
    values: list[str] = []

    def add_value(candidate: Any) -> None:
        if candidate is None:
            return
        if isinstance(candidate, (dict, list)):
            values.extend(collect_binding_values(candidate))
            return
        text = str(candidate).strip()
        if text and text != "-":
            values.append(text)

    if isinstance(value, dict):
        for key in (
            "value",
            "id",
            "label",
            "uri",
            "obj",
            "obj_uri",
            "class_uri",
            "prop_value",
            "object_uri",
            "object_name",
            "object",
            "subject_id",
            "subject_label",
            "subject",
            "object_id",
            "object_label",
            "o",
            "o_label",
            "c_o",
            "c_o_label",
            "s",
        ):
            if key in value:
                add_value(value[key])

        for key in (
            "types",
            "object_class",
            "o_class",
            "c_o_class",
            "fallback_classes",
            "important_entities",
        ):
            if key in value:
                add_value(value[key])

        if "attributes" in value:
            for attribute in value.get("attributes") or []:
                values.extend(collect_binding_values(attribute))

        for nested_key in (
            "results",
            "extracted_result",
            "extracted_results",
            "evidence",
            "relevant_entities",
            "parameter_values",
        ):
            if nested_key in value:
                values.extend(collect_binding_values(value[nested_key]))
        for key, nested_value in value.items():
            if key in {
                "p",
                "prop",
                "predicate",
                "predicate_id",
                "predicate_label",
                "object_description",
                "answer",
                "report",
                "sub_question",
                "original_question",
                "judge",
                "grounding",
                "token_usage",
                "calls",
            }:
                continue
            if isinstance(nested_value, (dict, list)):
                values.extend(collect_binding_values(nested_value))
        return values

    if isinstance(value, list):
        for item in value:
            values.extend(collect_binding_values(item))
        return values

    if isinstance(value, str):
        text = value.strip()
        return [text] if text and text != "-" else []

    return values


def extract_step_entities(step: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "important_entities",
        "parameter_values",
        "results",
        "extracted_result",
        "extracted_results",
        "evidence",
        "relevant_entities",
    ):
        values.extend(collect_binding_values(step.get(key)))
    return unique_entity_values(values)


def has_raw_sparql_binding_evidence(pred: dict[str, Any]) -> bool:
    evidence = pred.get("evidence") or []
    return any(
        isinstance(row, dict)
        and any(isinstance(value, dict) and "value" in value for value in row.values())
        for row in evidence
    )


def extract_retrieved_entities(pred: dict[str, Any], scope: str) -> list[str]:
    steps = pred.get("intermediary_results") or []
    if steps:
        selected_steps = [steps[-1]] if scope == "final" else steps
        values: list[str] = []
        for step in selected_steps:
            if isinstance(step, dict):
                values.extend(extract_step_entities(step))

        return unique_entity_values(values)

    if has_raw_sparql_binding_evidence(pred):
        return unique_entity_values(collect_binding_values(pred.get("evidence")))

    if scope == "final":
        values = collect_binding_values(pred.get("relevant_entities"))
        if not values:
            values = collect_binding_values(pred.get("evidence"))
        return unique_entity_values(values)

    values = []
    for key in ("relevant_entities", "evidence"):
        values.extend(collect_binding_values(pred.get(key)))
    return unique_entity_values(values)


def extract_bool_decision(text: Any) -> bool | None:
    normalized = normalize_text(text)
    if not normalized:
        return None

    false_patterns = (
        r"\bnot\s+(?:be\s+)?(?:generated|attributed|associated|connected|used)\b",
        r"\bwas\s+not\b",
        r"\bis\s+not\b",
        r"\bwere\s+not\b",
        r"\bno\b",
        r"\bfalse\b",
        r"\bnone\b",
    )
    true_patterns = (
        r"\bwas\s+(?:generated|attributed|associated|connected|used)\b",
        r"\bis\s+(?:generated|attributed|associated|connected|used)\b",
        r"\bwere\s+(?:generated|attributed|associated|connected|used)\b",
        r"\byes\b",
        r"\btrue\b",
    )

    if any(re.search(pattern, normalized) for pattern in false_patterns):
        return False
    if any(re.search(pattern, normalized) for pattern in true_patterns):
        return True
    return None


def extract_numeric_decision(text: Any) -> int | None:
    normalized = strip_citations(text)
    if not normalized:
        return None

    priority_patterns = (
        r"\b(?:is|are|equals?|count(?:s|ed)?|total(?:s)?)\D{0,80}(-?\d+)\b",
        r"\b(-?\d+)\s+(?:unique\s+)?(?:output|outputs|entities|execution|executions|records)\b",
    )
    for pattern in priority_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    integers = re.findall(r"(?<![A-Za-z0-9_])-?\d+(?![A-Za-z0-9_])", normalized)
    if not integers:
        return None
    return int(integers[0])


def metric_answer_token_overlap(
    pred: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
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


def metric_ground_truth_entity_coverage(
    pred: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
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


def metric_entity_retrieval(
    pred: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    gt_entities = extract_ground_truth_entity_values(actual)
    final_entities = extract_retrieved_entities(pred, scope="final")
    total_entities = extract_retrieved_entities(pred, scope="total")
    final_scores = retrieval_scores(gt_entities, final_entities)
    total_scores = retrieval_scores(gt_entities, total_entities)

    return {
        "entity_gt_total": len(entity_key_set(gt_entities)),
        "entity_retrieved_final_total": len(entity_key_set(final_entities)),
        "entity_retrieved_total_total": len(entity_key_set(total_entities)),
        "entity_recall_final": final_scores["recall"],
        "entity_precision_final": final_scores["precision"],
        "entity_f1_final": final_scores["f1"],
        "entity_recall_total": total_scores["recall"],
        "entity_precision_total": total_scores["precision"],
        "entity_f1_total": total_scores["f1"],
        "entity_gt_values": json.dumps(gt_entities, ensure_ascii=True),
        "entity_retrieved_final_values": json.dumps(final_entities, ensure_ascii=True),
        "entity_retrieved_total_values": json.dumps(total_entities, ensure_ascii=True),
    }


def metric_bool_accuracy(pred: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    gt_decision = actual.get("decision")
    pred_decision = extract_bool_decision(pred.get("answer", ""))
    accuracy = float(pred_decision == gt_decision) if gt_decision is not None else float("nan")

    return {
        "bool_ground_truth_decision": gt_decision,
        "bool_predicted_decision": pred_decision,
        "bool_accuracy": accuracy,
    }


def metric_numeric_accuracy(pred: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    gt_count = actual.get("count")
    pred_count = extract_numeric_decision(pred.get("answer", ""))
    accuracy = float(pred_count == gt_count) if gt_count is not None else float("nan")

    return {
        "numeric_ground_truth_count": gt_count,
        "numeric_predicted_count": pred_count,
        "numeric_accuracy": accuracy,
    }


METRIC_REGISTRY: dict[str, MetricFn] = {
    "answer_token_overlap": metric_answer_token_overlap,
    "ground_truth_entity_coverage": metric_ground_truth_entity_coverage,
    "entity_retrieval": metric_entity_retrieval,
    "bool_accuracy": metric_bool_accuracy,
    "numeric_accuracy": metric_numeric_accuracy,
}


def configured_metrics(
    configured_metric_names: Mapping[str, list[str]],
) -> dict[str, list[MetricFn]]:
    enabled_metrics: dict[str, list[MetricFn]] = {}
    for qtype, metric_names in configured_metric_names.items():
        filtered_names = [
            metric_name
            for metric_name in metric_names
            if metric_name not in REMOVED_METRICS
        ]
        for metric_name in sorted(FORCED_METRICS):
            if metric_name not in filtered_names:
                filtered_names.append(metric_name)
        enabled_metrics[qtype] = []
        for metric_name in filtered_names:
            if metric_name not in METRIC_REGISTRY:
                raise ValueError(f"Unknown or disabled evaluation metric: {metric_name}")
            enabled_metrics[qtype].append(METRIC_REGISTRY[metric_name])
    return enabled_metrics


def latest_prediction_file(
    experiments_dir: str | Path,
    prediction_filename: str = "RESULTS.jsonl",
) -> Path:
    experiments_path = resolve_repo_path(experiments_dir)
    if not experiments_path.exists():
        raise FileNotFoundError(f"Prediction directory not found: {experiments_path}")
    if not experiments_path.is_dir():
        raise NotADirectoryError(f"Prediction path is not a directory: {experiments_path}")

    experiment_dirs = sorted(
        (path for path in experiments_path.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for experiment_dir in experiment_dirs:
        prediction_file = experiment_dir / prediction_filename
        if prediction_file.exists():
            return prediction_file

    raise FileNotFoundError(
        f"No {prediction_filename} found in experiment folders under {experiments_path}"
    )


def resolve_prediction_files(
    prediction_dirs: Mapping[str, str | Path],
    prediction_files: Mapping[str, str | Path],
    prediction_filename: str,
    selected_methods: set[str] | None,
) -> dict[str, Path]:
    resolved = {
        codename: latest_prediction_file(experiments_dir, prediction_filename)
        for codename, experiments_dir in prediction_dirs.items()
        if selected_methods is None or codename in selected_methods
    }
    resolved.update(
        {
            codename: resolve_repo_path(file_path)
            for codename, file_path in prediction_files.items()
            if selected_methods is None or codename in selected_methods
        }
    )
    return resolved


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

    return {
        "evaluation_dir": evaluation_dir,
        "config_path": config_path,
        "config": config,
        "ground_truth_path": gt_path,
        "records": records,
        "by_id": {record["id"]: record for record in records},
        "by_question": {
            normalize_question(record["question"]): record
            for record in records
        },
    }


def ours_input_config(
    record: dict[str, Any],
    synthetic_question_retriever: SQRetriver,
    answer_report: str,
) -> dict[str, Any]:
    return augment_ours_record_with_reports(
        record,
        synthetic_question_retriever=synthetic_question_retriever,
        answer_report=answer_report,
    )


def grasp_input_config(record: dict[str, Any]) -> dict[str, Any]:
    output = record.get("output", {})

    question = ""
    for message in record.get("messages", []):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                question = content
                break

    if not output:
        return {
            "_line_number": record["_line_number"],
            "_prediction_path": record["_prediction_path"],
            "answer": "",
            "evidence": [{"sparql_error": "Output Null"}],
            "id": record.get("id"),
            "question": question,
            "relevant_entities": [],
            "time_taken": record.get("elapsed"),
        }

    endpoint = output.get("endpoint", "http://localhost:3030/ds/sparql")
    sparql_query = output.get("sparql")
    evidence: list[dict[str, Any]] = []
    relevant_entities: list[str] = []

    if endpoint and sparql_query:
        try:
            req = requests.post(endpoint, data={"query": sparql_query}, timeout=30)
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
            error_body = (
                (exc.response.text or "").strip()
                if exc.response is not None
                else ""
            )
            status_code = (
                exc.response.status_code
                if exc.response is not None
                else "unknown"
            )
            evidence = [{"sparql_error": f"HTTP {status_code}: {error_body or str(exc)}"}]
        except requests.RequestException as exc:
            evidence = [{"sparql_error": str(exc)}]

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


def build_input_augmentation_map(
    selected_methods: set[str] | None,
    settings: Mapping[str, Any],
    answer_report: str,
) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    augmentations: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
    if selected_methods is None or "grasp" in selected_methods:
        augmentations["grasp"] = grasp_input_config
    if selected_methods is None or "ours" in selected_methods:
        sq_loc = settings.get("sq_loc")
        if sq_loc:
            sq_retriever = SQRetriver(str(resolve_repo_path(sq_loc)))
            augmentations["ours"] = partial(
                ours_input_config,
                synthetic_question_retriever=sq_retriever,
                answer_report=answer_report,
            )
    return augmentations


def read_jsonl(path_value: str | Path) -> list[dict[str, Any]]:
    path = resolve_repo_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    records: list[dict[str, Any]] = []
    for line_number, record in enumerate(
        common_utils.serialization.load_jsonl(path),
        start=1,
    ):
        record["_prediction_path"] = str(path)
        record["_line_number"] = line_number
        records.append(record)
    return records


def load_prediction_runs(
    prediction_files: Mapping[str, str | Path],
    data_augmentations: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]],
    max_examples_per_run: int | None,
) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    for codename, file_path in prediction_files.items():
        records = read_jsonl(file_path)
        if max_examples_per_run is not None:
            records = records[:max_examples_per_run]

        if codename in data_augmentations:
            records = [data_augmentations[codename](record) for record in records]
        runs[codename] = records
    return runs


def resolve_ground_truth_record(
    pred: dict[str, Any],
    gt_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    pred_id = pred.get("id")
    if pred_id in gt_bundle["by_id"]:
        return gt_bundle["by_id"][pred_id]

    question_key = normalize_question(pred.get("question", ""))
    gt = gt_bundle["by_question"].get(question_key)
    if not gt:
        raise ValueError(f"No GT found for prediction: {pred.get('id')}")
    return gt


def evaluate_run(
    run_name: str,
    predictions: list[dict[str, Any]],
    metric_functions: Mapping[str, list[MetricFn]],
    gt_bundle: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_category: dict[str, list[dict[str, Any]]] = {}

    for pred in tqdm(predictions, desc=run_name):
        actual = resolve_ground_truth_record(pred, gt_bundle)
        base_row: dict[str, Any] = {
            "run": run_name,
            "prediction_path": pred.get("_prediction_path"),
            "prediction_line": pred.get("_line_number"),
            "prediction_id": pred.get("id"),
            "ground_truth_found": actual is not None,
            "question": pred.get("question", ""),
            "pred_answer": strip_citations(pred.get("answer", "")),
            "prediction_time_taken": pred.get("time_taken"),
            "ground_truth_id": actual.get("id"),
            "ground_truth_qtype": json.dumps(actual.get("qtype", []), ensure_ascii=True),
            "ground_truth_tags": json.dumps(
                actual.get("tags", {}),
                ensure_ascii=True,
                sort_keys=True,
            ),
            "ground_truth_answer": actual.get("answer", ""),
        }

        for category, functions in metric_functions.items():
            category_tags = category.split("--")
            actual_qtypes = actual.get("qtype") or []
            if not actual_qtypes:
                raise ValueError(f"qtype is empty for question: {pred.get('id')}")
            if set(category_tags) - set(actual_qtypes):
                continue

            row = dict(base_row)
            for metric_fn in functions:
                try:
                    row.update(metric_fn(pred, actual))
                except Exception as exc:
                    row[f"{metric_fn.__name__}_error"] = str(exc)

            rows_by_category.setdefault(category, []).append(row)

    return rows_by_category


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


def sort_results_df(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    sort_columns = [
        column
        for column in ("run", "ground_truth_found", "prediction_line")
        if column in results_df.columns
    ]
    if not sort_columns:
        return results_df.reset_index(drop=True)

    return results_df.sort_values(
        sort_columns,
        ascending=[True, False, True][: len(sort_columns)],
    ).reset_index(drop=True)


def build_overall_summary(results_by_category: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    category_frames = [
        df.assign(category=category)
        for category, df in results_by_category.items()
        if not df.empty
    ]
    if not category_frames:
        return pd.DataFrame(columns=["run", "evaluated_examples", *OVERALL_METRIC_COLUMNS])

    all_results = pd.concat(category_frames, ignore_index=True, sort=False)
    for column in OVERALL_METRIC_COLUMNS:
        if column in all_results.columns:
            all_results[column] = pd.to_numeric(all_results[column], errors="coerce")

    counts_df = (
        all_results.groupby("run", dropna=False)
        .agg(
            evaluated_examples=("run", "size"),
            matched_ground_truth=("ground_truth_found", "sum"),
        )
        .reset_index()
    )
    available_metric_columns = [
        column for column in OVERALL_METRIC_COLUMNS if column in all_results.columns
    ]
    if not available_metric_columns:
        return counts_df

    metric_df = (
        all_results.groupby("run", dropna=False)[available_metric_columns]
        .mean()
        .reset_index()
    )
    return counts_df.merge(metric_df, on="run", how="left")


def order_method_rows(table: pd.DataFrame, method_column: str = "Method") -> pd.DataFrame:
    if method_column not in table.columns:
        return table

    ordered = table.copy()
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    fallback_rank = len(method_rank)
    ordered["_method_order"] = ordered[method_column].map(method_rank).fillna(fallback_rank)
    ordered = ordered.sort_values(["_method_order", method_column]).drop(
        columns=["_method_order"]
    )
    return ordered.reset_index(drop=True)


def scale_result_columns(table: pd.DataFrame) -> pd.DataFrame:
    scaled = table.copy()
    for column in scaled.columns:
        if column in COUNT_COLUMNS:
            continue
        values = scaled[column]
        if pd.api.types.is_numeric_dtype(values):
            scaled[column] = values * 100
    return scaled


def format_overall_table(overall_summary_df: pd.DataFrame, round_digits: int) -> pd.DataFrame:
    if overall_summary_df.empty:
        return overall_summary_df.rename(columns=COLUMN_LABELS)

    display_columns = [
        "run",
        "evaluated_examples",
        "matched_ground_truth",
        *[column for column in OVERALL_METRIC_COLUMNS if column in overall_summary_df.columns],
    ]
    table = overall_summary_df[display_columns].copy()
    table["run"] = table["run"].replace(METHOD_LABELS)
    table = table.rename(columns=COLUMN_LABELS)
    table = scale_result_columns(table).round(round_digits)
    return order_method_rows(table)


def prepare_output_dir(save_dir: Path, overwrite: bool) -> Path:
    if overwrite or not save_dir.exists() or not any(save_dir.iterdir()):
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = save_dir / f"run_{timestamp}"
    timestamped_dir.mkdir(parents=True, exist_ok=False)
    return timestamped_dir


def write_outputs(
    output_dir: Path,
    results_by_category: Mapping[str, pd.DataFrame],
    summaries_by_category: Mapping[str, pd.DataFrame],
    overall_table: pd.DataFrame,
) -> None:
    for category, results_df in results_by_category.items():
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(category_dir / "results.csv", index=False)
        summaries_by_category[category].to_csv(
            category_dir / "results_summary.csv",
            index=False,
        )
    overall_table.to_csv(output_dir / "overall_summary.csv", index=False)


def print_prediction_inventory(prediction_runs: Mapping[str, list[dict[str, Any]]]) -> None:
    inventory = pd.DataFrame(
        [
            {
                "run": codename,
                "file": records[0]["_prediction_path"] if records else "",
                "rows": len(records),
            }
            for codename, records in prediction_runs.items()
        ]
    )
    print("\nPrediction inventory")
    print(inventory.to_string(index=False))


def print_overall_table(overall_table: pd.DataFrame) -> None:
    print("\nOverall summary")
    if overall_table.empty:
        print("No rows were evaluated.")
    else:
        print(overall_table.to_string(index=False))


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")

    evaluation_dir = resolve_repo_path(Path("evaluations") / args.evaluation)
    config_path = evaluation_dir / "config.evaluation.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")

    evaluation_config = load_config(str(config_path))
    settings = evaluation_config.get("evaluation", {})
    metrics_config = settings.get("metrics", {})
    metrics_by_qtype = configured_metrics(
        metrics_config.get("enabled_by_qtype", {}),
    )

    prediction_dirs = dict(settings.get("prediction_dirs", {}))
    configured_prediction_files = dict(settings.get("prediction_files", {}))
    available_methods = set(prediction_dirs) | set(configured_prediction_files)
    selected_methods = set(args.methods) if args.methods else None
    if selected_methods:
        unknown_methods = selected_methods - available_methods
        if unknown_methods:
            raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")

    prediction_files = resolve_prediction_files(
        prediction_dirs,
        configured_prediction_files,
        settings.get("prediction_filename", "RESULTS.jsonl"),
        selected_methods,
    )

    answer_report = args.answer_report or metrics_config.get(
        "answer_report",
        settings.get("answer_report", "original"),
    )
    input_augmentations = build_input_augmentation_map(
        selected_methods,
        settings,
        answer_report,
    )

    gt_bundle = load_ground_truth_bundle(
        args.evaluation,
        settings.get("source_config", "config.fullcontext.yaml"),
    )

    print(f"Config: {config_path}")
    print(f"Ground truth: {gt_bundle['ground_truth_path']}")
    print(f"Ground-truth examples: {len(gt_bundle['records'])}")
    print(f"Enabled qtype metrics: {', '.join(sorted(metrics_by_qtype))}")
    print("Removed metrics: bert_score, llm_answer_quality, nli_entailment")
    print("Forced metric on every qtype: entity_retrieval")

    prediction_runs = load_prediction_runs(
        prediction_files,
        input_augmentations,
        args.max_examples_per_run,
    )
    print_prediction_inventory(prediction_runs)

    evaluation_rows: dict[str, list[dict[str, Any]]] = {}
    for run_name, predictions in prediction_runs.items():
        run_rows = evaluate_run(run_name, predictions, metrics_by_qtype, gt_bundle)
        for category, rows in run_rows.items():
            evaluation_rows.setdefault(category, []).extend(rows)

    categories_to_write = sorted(set(metrics_by_qtype) | set(evaluation_rows))
    results_by_category: dict[str, pd.DataFrame] = {}
    summaries_by_category: dict[str, pd.DataFrame] = {}
    for category in categories_to_write:
        results_df = sort_results_df(pd.DataFrame(evaluation_rows.get(category, [])))
        summary_df = build_run_summary(results_df)
        results_by_category[category] = results_df
        summaries_by_category[category] = summary_df
        print(f"{category}: {len(results_df)} evaluated rows")

    overall_summary_df = build_overall_summary(results_by_category)
    overall_table = format_overall_table(overall_summary_df, args.round_digits)
    print_overall_table(overall_table)

    if not args.no_save:
        default_save_dir = evaluation_dir / "analysis2"
        requested_save_dir = resolve_repo_path(args.save_dir) if args.save_dir else default_save_dir
        output_dir = prepare_output_dir(requested_save_dir, overwrite=args.overwrite)
        write_outputs(output_dir, results_by_category, summaries_by_category, overall_table)
        print(f"\nWrote outputs under: {output_dir}")


if __name__ == "__main__":
    main()
