# %%
from pathlib import Path
import argparse
import asyncio
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, Dict, List
import sys

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
_nli_model: Any | None = None


def get_judge_llm() -> Any:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = LLM(
            JUDGE_LLM["llm_type"],
            JUDGE_LLM.get("llm_library", "dspy"),
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
            faithfulness=float(scores.faithfulness),
            relevance=float(scores.relevance),
            understanderbility=float(scores.understanderbility),
        )
        return {
            "llm_completeness": validated_scores.completeness,
            "llm_faithfulness": validated_scores.faithfulness,
            "llm_relevance": validated_scores.relevance,
            "llm_understanderbility": validated_scores.understanderbility,
        }

    metric.__name__ = "llm_answer_quality"
    return metric

def build_nli_premises(pred: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    report = strip_citations(pred.get("answer", ""))
    return [report] if report else []


def build_nli_hypotheses(actual: dict[str, Any]) -> list[str]:
    question = str(actual.get("question", "") or "").strip()
    answer = strip_citations(actual.get("answer", ""))
    hypothesis = f"Question: {question}\nAnswer: {answer}".strip()
    return [hypothesis] if hypothesis else []


def get_nli_model() -> Any:
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder

        _nli_model = CrossEncoder(NLI_MODEL_NAME)
    return _nli_model


def get_nli_entailment_index(model: Any) -> int:
    id2label = getattr(getattr(model, "model", None), "config", None)
    id2label = getattr(id2label, "id2label", {}) or {}
    for label_index, label in id2label.items():
        if "entail" in str(label).lower():
            return int(label_index)
    return NLI_ENTAILMENT_FALLBACK_INDEX


def score_nli_entailment(premise: str, hypothesis: str) -> float:
    model = get_nli_model()
    scores = model.predict([(premise, hypothesis)], apply_softmax=True)
    entailment_index = get_nli_entailment_index(model)
    label_scores = scores[0] if getattr(scores, "ndim", 1) > 1 else scores
    return float(label_scores[entailment_index])


def metric_nli_entailment(pred: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    premises = build_nli_premises(pred, actual)
    hypotheses = build_nli_hypotheses(actual)
    best_score = float("nan")
    best_premise = ""
    best_hypothesis = ""
    scored_pairs = 0

    for premise in premises:
        for hypothesis in hypotheses:
            scored_pairs += 1
            entailment_score = score_nli_entailment(premise, hypothesis)
            if scored_pairs == 1 or entailment_score > best_score:
                best_score = entailment_score
                best_premise = premise
                best_hypothesis = hypothesis

    return {
        "nli_entailment_max": best_score,
        "nli_pairs_scored": scored_pairs,
        "nli_best_premise": best_premise,
        "nli_best_hypothesis": best_hypothesis,
    }


def metric_entity_retrieval(pred: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
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


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _get_evaluation_choices() -> list[str]:
    evaluations_dir = Path(__file__).resolve().parent / "evaluations"
    return sorted(
        path.name
        for path in evaluations_dir.iterdir()
        if path.is_dir() and path.name != "test_questions"
    )


def _parse_config_path() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--evaluation",
        choices=_get_evaluation_choices(),
        default="chatbs-base",
        help="Evaluation folder under evaluations/ to load config from.",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return str(Path("evaluations") / args.evaluation / "config.evaluation.yaml")


CONFIG_PATH = _parse_config_path()
EVALUATION_CONFIG = load_config(CONFIG_PATH)
EVALUATION_SETTINGS = EVALUATION_CONFIG.get("evaluation", {})

METRIC_REGISTRY: dict[str, MetricFn] = {
    "answer_token_overlap": metric_answer_token_overlap,
    "ground_truth_entity_coverage": metric_ground_truth_entity_coverage,
    "bert_score": metric_bert_score,
    "nli_entailment": metric_nli_entailment,
    "entity_retrieval": metric_entity_retrieval,
    "bool_accuracy": metric_bool_accuracy,
    "numeric_accuracy": metric_numeric_accuracy,
    "llm_answer_quality": build_llm_answer_quality_metric(),
}


def configured_metrics(configured_metric_names: Mapping[str, list[str]]) -> Dict[str, List[MetricFn]]:
    enabled_metrics: Dict[str, List[MetricFn]] = {}
    for qtype, metric_names in configured_metric_names.items():
        enabled_metrics[qtype] = []
        for metric_name in metric_names:
            if metric_name not in METRIC_REGISTRY:
                raise ValueError(f"Unknown evaluation metric in config: {metric_name}")
            enabled_metrics[qtype].append(METRIC_REGISTRY[metric_name])
    return enabled_metrics


EVALUATION_NAME = EVALUATION_SETTINGS.get("name", "chatbs-base")
CONFIG_FILENAME = EVALUATION_SETTINGS.get("source_config", "config.fullcontext.yaml")
PREDICTION_FILENAME = EVALUATION_SETTINGS.get("prediction_filename", "RESULTS.jsonl")
PREDICTION_DIRS = dict(EVALUATION_SETTINGS.get("prediction_dirs", {}))
CONFIGURED_PREDICTION_FILES = dict(EVALUATION_SETTINGS.get("prediction_files", {}))
JUDGE_LLM = dict(EVALUATION_SETTINGS.get("judge_llm", {}))
NLI_CONFIG = EVALUATION_SETTINGS.get("nli", {})
NLI_MODEL_NAME = NLI_CONFIG.get("model_name", "cross-encoder/nli-distilroberta-base")
NLI_ENTAILMENT_FALLBACK_INDEX = NLI_CONFIG.get("entailment_fallback_index", 1)
ENABLED_METRICS = configured_metrics(
    EVALUATION_SETTINGS.get("metrics", {}).get("enabled_by_qtype", {})
)
MAX_EXAMPLES_PER_RUN = EVALUATION_SETTINGS.get("max_examples_per_run")
SAVE_DIR = resolve_repo_path(
    EVALUATION_SETTINGS.get(
        "save_dir",
        str(Path("evaluations") / EVALUATION_NAME / "analysis"),
    )
)


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
    prediction_filename: str = "RESULTS.jsonl",
) -> dict[str, Path]:
    resolved = {
        codename: latest_prediction_file(experiments_dir, prediction_filename)
        for codename, experiments_dir in prediction_dirs.items()
    }
    resolved.update(
        {
            codename: resolve_repo_path(file_path)
            for codename, file_path in prediction_files.items()
        }
    )
    return resolved


PREDICTION_FILES = resolve_prediction_files(
    PREDICTION_DIRS,
    CONFIGURED_PREDICTION_FILES,
    PREDICTION_FILENAME,
)


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

from src.synthetic_questions import SQRetriver

GT_BUNDLE = load_ground_truth_bundle(EVALUATION_NAME, CONFIG_FILENAME)
sq_retriver = SQRetriver(
    EVALUATION_SETTINGS.get("sq_loc")
)

print(f"Config: {GT_BUNDLE['config_path']}")
print(f"Ground truth: {GT_BUNDLE['ground_truth_path']}")
print(f"Ground-truth examples: {len(GT_BUNDLE['records'])}")

display(
    pd.DataFrame(GT_BUNDLE["records"])[["id", "question", "qtype"]].head()
)

# %%
def attribute_display(uri, attr):
    entity_rep = ["{}[{}]".format(uri, attr['rdf:type'])]
    for k,v in attr.items:
        if k in ["rdf:type"]:
            continue
        
        line = "\t{} -> {}".format(k,v["object"])
        line += "[{}]".format(v['object_class']) if "-" != v['object_class'] else ""
        
        if "-" != v['object_label']:
            if len(v['object_label'])>20:
                lbl = v['object_label'][:20]+" ..."
            else:
                lbl = v['object_label']
            line += "({})".format(lbl)
        entity_rep.append(
            line
        )
        
    return "\n".join(entity_rep)


def build_trace_report(record:Dict[str, Any], max_ent_per_step=3) -> str:  
    blocks = ["User Question:{}".format(record['question']),
              "Trace Answers"]
    for step in record.get('intermediary_results', []):
        step_rep = []
        step_rep.append(
                "Question: {}".format(step.get('sub_question', ""))
            )
        
        if step['strategy'] == 'by_program':
            program = sq_retriver.get_program_by_id(
                step["program_id"]
            )
            
            if program:
                step_rep.append(
                    "KG Question: {}".format(program["solves"]),
                )
                
        elif step['strategy'] == "by_linked_data":
            step_rep.append(
                    "Linked Entities:",
                )
            
        step_rep.append(
                    "Result Entities:\n{}".format(
                        ", ".join(step["important_entities"])
                        )
                )
        
        extracted_entities = step.get("extracted_entities", [])
        if extracted_entities:
            step_rep.append(
                    "Example Entities:"
                )

            ent_types = {}
            for ent in extracted_entities:
                attr_dict = {v['relation']:v  for v in ent["attributes"]}
                if attr_dict['rdf:type'] in ent_types:
                    ent_types[attr_dict['rdf:type']]["ent_count"] += 1
                    ent_types[attr_dict['rdf:type']]["attr_count"] = max(
                        ent_types[attr_dict['rdf:type']]["attr_count"],
                        len(attr_dict)
                    )
                else:
                    ent_types[attr_dict['rdf:type']] = {
                        "ent_count":1,
                        "attr_count":len(attr_dict)
                    }
                    
            if len(ent_types) == 1:
                _cls = list(ent_types.keys())[0]
                if ent_types[_cls]["attr_count"] > 5:
                    ent = extracted_entities[0]
                    attr_dict = {v['relation']:v  for v in ent["attributes"]}
                    step_rep.append(
                        attribute_display(
                            ent['uri'], attr_dict
                        )
                    )
                else:
                    ents_sel = extracted_entities[:max_ent_per_step]
                    for ent in ents_sel:
                        attr_dict = {v['relation']:v  for v in ent["attributes"]}
                        step_rep.append(
                            attribute_display(
                                ent['uri'], attr_dict
                            )
                        )
            else:
                already_visited = set()
                for ent in extracted_entities:
                    attr_dict = {v['relation']:v  for v in ent["attributes"]}
                    if attr_dict["rdf:type"] not in already_visited:
                        step_rep.append(
                            attribute_display(
                                ent['uri'], attr_dict
                            )
                        )
                        
                    already_visited.add(attr_dict["rdf:type"])    
                    
        blocks.append(
            "\n".join(step_rep)
        )
        
    blocks.append(
        "Summary Answer:\n{}".format(
            record.get("answer", "")
        )
    )            
    
    return "\n\n".join(blocks)

def build_eo_trace_report(record:Dict[str, Any]) -> str:
    print(record)
    
    """
    ### Knowledge Based System: 
    {system_name}

    ### What were the system outputs associated with the user query and the system trace?: 
    (System Recommendation)
    {system_recommendation}

    ### What are the entities associate with the question: 
    (wasGeneratedBy)
    {generated_by}

    ### Traces Associated with the system recommendation:
    (System Trace)
    {system_trace}
    
    ### Overall Answer to the Question:
    {overall_answer}
    """
    return ""


def ours_input_config(record:Dict[str, Any]) -> Dict[str, Any]:
    record["report"] = build_trace_report(record)
    record["eo_report"] = build_eo_trace_report(record)
    record["answer"] = record["report"]
    
    return record

# %%
def grasp_input_config(record:Dict[str, Any]) -> Dict[str, Any]:
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
    "grasp":grasp_input_config,
    "ours": ours_input_config
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


def clean_entity_value(value: Any) -> str:
    text = strip_citations(value)
    text = re.sub(r"@[a-zA-Z-]+(?:\^\^<[^>]+>)?$", "", text)
    text = re.sub(r"\^\^<[^>]+>$", "", text)
    text = text.strip().strip("<>").strip()
    return text


def entity_aliases(value: Any) -> set[str]:
    text = clean_entity_value(value)
    if not text:
        return set()

    aliases = {normalize_text(text)}
    compact_text = text.replace("http://testwebsite/testProgram#", "ChatBS-NexGen:")
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

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


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


def collect_binding_values(value: Any) -> list[str]:
    values: list[str] = []

    if isinstance(value, dict):
        if "value" in value and value["value"] is not None:
            values.append(str(value["value"]))

        for key in (
            "id",
            "label",
            "uri",
            "object_uri",
            "object_name",
            "subject_id",
            "subject_label",
            "object_id",
            "object_label",
        ):
            if key in value and value[key] is not None:
                values.append(str(value[key]))

        if "types" in value:
            values.extend(str(item) for item in value.get("types") or [] if item is not None)
        if "object_class" in value:
            values.extend(str(item) for item in value.get("object_class") or [] if item is not None)
        if "important_entities" in value:
            values.extend(str(item) for item in value.get("important_entities") or [] if item is not None)

        if "attributes" in value:
            for attribute in value.get("attributes") or []:
                if isinstance(attribute, dict):
                    for key in ("s", "o", "value"):
                        if attribute.get(key) is not None:
                            values.append(str(attribute[key]))

        for nested_key in ("results", "extracted_results", "evidence", "relevant_entities"):
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
            }:
                continue
            if isinstance(nested_value, (dict, list)):
                values.extend(collect_binding_values(nested_value))
        return values

    if isinstance(value, list):
        for item in value:
            values.extend(collect_binding_values(item))
        return values

    if isinstance(value, str) and value.strip():
        return [value]

    return values


def extract_step_entities(step: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("important_entities", "results", "extracted_results", "evidence", "relevant_entities"):
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
                ValueError(f"tags is None for question: {pred.get('id')}")
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
