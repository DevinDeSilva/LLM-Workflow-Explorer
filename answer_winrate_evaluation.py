# %%
from pathlib import Path
import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from itertools import combinations
from typing import Any, Dict

import dspy
import dycomutils as common_utils
import pandas as pd
import requests
from dotenv import load_dotenv
from IPython.display import display
from tqdm import tqdm

from src.config.experiment import FullContextExperimentConfig
from src.experiment.ground_truth import GTInfo
from src.llm import LLM
from src.utils.utils import load_config

REPO_ROOT = Path.cwd()
load_dotenv(REPO_ROOT / ".env")

_winrate_judge_llm: Any | None = None


class PairwiseAnswerWinrateSignature(dspy.Signature):
    """
    You are a evaluator and your given outputs of two explanation generation systems for AI
    Systems. Your job is to identify which method provides a better explanation to the asked question.
    A human curated answer is also provided.

    Choose which answer is more understandable by a lay user.
    Be cautious whether the answer is faithful to the human curated answer.
    """

    question: str = dspy.InputField()
    ground_truth_answer: str = dspy.InputField()
    method_a: str = dspy.InputField()
    answer_a: str = dspy.InputField()
    method_b: str = dspy.InputField()
    answer_b: str = dspy.InputField()
    winner: str = dspy.OutputField(
        desc="Return exactly one of: method_a, method_b, tie."
    )
    rationale: str = dspy.OutputField(
        desc="Brief reason for the choice, grounded in the reference answer."
    )


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

EVALUATION_NAME = EVALUATION_SETTINGS.get("name", "chatbs-base")
CONFIG_FILENAME = EVALUATION_SETTINGS.get("source_config", "config.fullcontext.yaml")
PREDICTION_FILENAME = EVALUATION_SETTINGS.get("prediction_filename", "RESULTS.jsonl")
PREDICTION_DIRS = dict(EVALUATION_SETTINGS.get("prediction_dirs", {}))
CONFIGURED_PREDICTION_FILES = dict(EVALUATION_SETTINGS.get("prediction_files", {}))
JUDGE_LLM = dict(EVALUATION_SETTINGS.get("judge_llm", {}))
MAX_EXAMPLES_PER_RUN = EVALUATION_SETTINGS.get("max_examples_per_run")
SAVE_DIR = resolve_repo_path(
    EVALUATION_SETTINGS.get(
        "save_dir",
        str(Path("evaluations") / EVALUATION_NAME / "analysis"),
    )
)
WINRATE_CONFIG = EVALUATION_SETTINGS.get("winrate", {})
WINRATE_JUDGE_LLM = WINRATE_CONFIG.get("judge_llm") or JUDGE_LLM
WINRATE_TIE_SCORE = WINRATE_CONFIG.get("tie_score", 0.5)
WINRATE_SAVE_DIR = SAVE_DIR / "answer_winrate"


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
WINRATE_METHODS = WINRATE_CONFIG.get("methods") or list(PREDICTION_FILES.keys())


def get_winrate_judge_llm() -> Any:
    global _winrate_judge_llm
    if _winrate_judge_llm is None:
        _winrate_judge_llm = LLM(
            WINRATE_JUDGE_LLM["llm_type"],
            WINRATE_JUDGE_LLM.get("llm_library", "dspy"),
            **WINRATE_JUDGE_LLM["llm_config"],
        )
    return _winrate_judge_llm


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
print(f"Ground-truth examples: {len(GT_BUNDLE['records'])}")

display(
    pd.DataFrame(GT_BUNDLE["records"])[["id", "question", "qtype"]].head()
)


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
    "grasp": grasp_input_config,
}


def read_jsonl(path_value: str | Path) -> list[dict[str, Any]]:
    path = resolve_repo_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    records: list[dict[str, Any]] = []

    for line_number, record in enumerate(common_utils.serialization.load_jsonl(path), start=1):
        record["_prediction_path"] = str(path)
        record["_line_number"] = line_number
        records.append(record)

    if len(records) > 1:
        print(f"{str(path).split(os.sep)[-3]}\n{sorted(records[0].keys())}\n")
    return records


def load_prediction_runs(
    prediction_files: Mapping[str, str | Path],
    data_augmentations: dict[str, Callable],
) -> dict[str, list[dict[str, Any]]]:
    if not prediction_files:
        raise ValueError("Add at least one prediction path to config.evaluation.yaml.")

    runs: dict[str, list[dict[str, Any]]] = {}
    for codename, file_path in prediction_files.items():
        records = read_jsonl(file_path)
        if MAX_EXAMPLES_PER_RUN is not None:
            records = records[:MAX_EXAMPLES_PER_RUN]

        runs[codename] = records
        if codename in data_augmentations:
            for i, record in enumerate(runs[codename]):
                runs[codename][i] = data_augmentations[codename](record)
    return runs


PREDICTION_RUNS = load_prediction_runs(
    PREDICTION_FILES,
    INPUT_AUGMENTATION_MAP,
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


def resolve_ground_truth_record(pred: dict[str, Any]) -> Dict[str, Any] | None:
    pred_id = pred.get("id")
    if pred_id in GT_BUNDLE["by_id"]:
        return GT_BUNDLE["by_id"][pred_id]

    question_key = normalize_question(pred.get("question", ""))
    gt = GT_BUNDLE["by_question"].get(question_key)

    if not gt:
        raise ValueError(f"No GT found for prediction: {pred.get('id')}")

    return gt


def build_prediction_index_by_ground_truth(
    prediction_runs: Mapping[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
    prediction_index: dict[str, dict[str, dict[str, Any]]] = {}
    ground_truth_index: dict[str, dict[str, Any]] = {}

    for method in WINRATE_METHODS:
        method_index: dict[str, dict[str, Any]] = {}
        for pred in prediction_runs.get(method, []):
            actual = resolve_ground_truth_record(pred)
            if actual is None:
                continue

            ground_truth_id = actual.get("id")
            if not ground_truth_id or ground_truth_id in method_index:
                continue

            method_index[ground_truth_id] = pred
            ground_truth_index[ground_truth_id] = actual

        prediction_index[method] = method_index

    return prediction_index, ground_truth_index


def normalize_pairwise_winner(
    winner: Any,
    method_a: str | None = None,
    method_b: str | None = None,
) -> str:
    normalized = normalize_text(winner)
    normalized_method_a = normalize_text(method_a)
    normalized_method_b = normalize_text(method_b)
    if normalized_method_a and normalized == normalized_method_a:
        return "method_a"
    if normalized_method_b and normalized == normalized_method_b:
        return "method_b"
    if normalized in {"method a", "a", "answer a", "output 1", "1"}:
        return "method_a"
    if normalized in {"method b", "b", "answer b", "output 2", "2"}:
        return "method_b"
    if normalized in {"tie", "same", "draw", "equal", "equivalent", "0"}:
        return "tie"
    if "method a" in normalized:
        return "method_a"
    if "method b" in normalized:
        return "method_b"
    if "tie" in normalized or "same" in normalized or "equal" in normalized:
        return "tie"
    return "error"


def pairwise_answer_scores(winner: str) -> tuple[float, float, str | None]:
    if winner == "method_a":
        return 1.0, 0.0, "method_a"
    if winner == "method_b":
        return 0.0, 1.0, "method_b"
    if winner == "tie":
        return WINRATE_TIE_SCORE, WINRATE_TIE_SCORE, None
    return float("nan"), float("nan"), None


def run_pairwise_answer_winrate(
    prediction_runs: Mapping[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    methods = [method for method in WINRATE_METHODS if method in prediction_runs]
    if len(methods) < 2:
        return pd.DataFrame()

    prediction_index, ground_truth_index = build_prediction_index_by_ground_truth(
        prediction_runs
    )
    judge = dspy.Predict(PairwiseAnswerWinrateSignature)
    rows: list[dict[str, Any]] = []

    for method_a, method_b in combinations(methods, 2):
        if (method_b != "ours") and (method_a != "ours"):
            continue

        shared_ground_truth_ids = sorted(
            set(prediction_index.get(method_a, {}))
            & set(prediction_index.get(method_b, {}))
        )
        if MAX_EXAMPLES_PER_RUN is not None:
            shared_ground_truth_ids = shared_ground_truth_ids[:MAX_EXAMPLES_PER_RUN]

        pair_label = f"{method_a} vs {method_b}"
        for ground_truth_id in tqdm(shared_ground_truth_ids, desc=pair_label):
            actual = ground_truth_index[ground_truth_id]
            pred_a = prediction_index[method_a][ground_truth_id]
            pred_b = prediction_index[method_b][ground_truth_id]
            answer_a = strip_citations(pred_a.get("answer", ""))
            answer_b = strip_citations(pred_b.get("answer", ""))

            row: dict[str, Any] = {
                "ground_truth_id": ground_truth_id,
                "ground_truth_qtype": json.dumps(actual.get("qtype", []), ensure_ascii=True),
                "question": actual.get("question", ""),
                "ground_truth_answer": actual.get("answer", ""),
                "method_a": method_a,
                "method_b": method_b,
                "answer_a": answer_a,
                "answer_b": answer_b,
            }

            try:
                judge_llm = get_winrate_judge_llm()
                with dspy.context(lm=judge_llm.llm):
                    preference = judge(
                        question=actual.get("question", ""),
                        ground_truth_answer=actual.get("answer", ""),
                        method_a=method_a,
                        answer_a=answer_a,
                        method_b=method_b,
                        answer_b=answer_b,
                    )

                winner = normalize_pairwise_winner(
                    preference.winner,
                    method_a,
                    method_b,
                )
                row["winner"] = winner
                row["judge_rationale"] = getattr(preference, "rationale", "")
            except Exception as exc:
                winner = "error"
                row["winner"] = winner
                row["judge_error"] = str(exc)

            method_a_score, method_b_score, winning_side = pairwise_answer_scores(winner)
            row["winning_method"] = (
                row[winning_side] if winning_side is not None else None
            )
            row["method_a_score"] = method_a_score
            row["method_b_score"] = method_b_score
            rows.append(row)

    return pd.DataFrame(rows)


def build_answer_winrate_summary(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    if pairwise_df.empty:
        return pd.DataFrame()

    valid_df = pairwise_df[pairwise_df["winner"].isin(["method_a", "method_b", "tie"])]
    if valid_df.empty:
        return pd.DataFrame()

    method_rows: list[dict[str, Any]] = []
    for row in valid_df.to_dict("records"):
        method_rows.append(
            {
                "method": row["method_a"],
                "opponent": row["method_b"],
                "score": row["method_a_score"],
                "win": int(row["winner"] == "method_a"),
                "loss": int(row["winner"] == "method_b"),
                "tie": int(row["winner"] == "tie"),
            }
        )
        method_rows.append(
            {
                "method": row["method_b"],
                "opponent": row["method_a"],
                "score": row["method_b_score"],
                "win": int(row["winner"] == "method_b"),
                "loss": int(row["winner"] == "method_a"),
                "tie": int(row["winner"] == "tie"),
            }
        )

    method_df = pd.DataFrame(method_rows)
    summary_df = (
        method_df.groupby("method", dropna=False)
        .agg(
            comparisons=("score", "size"),
            wins=("win", "sum"),
            losses=("loss", "sum"),
            ties=("tie", "sum"),
            winrate=("score", "mean"),
        )
        .reset_index()
        .sort_values(["winrate", "wins", "ties"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    return summary_df


def main() -> None:
    answer_winrate_df = run_pairwise_answer_winrate(PREDICTION_RUNS)
    answer_winrate_summary_df = build_answer_winrate_summary(answer_winrate_df)

    display(answer_winrate_df.head())
    display(answer_winrate_summary_df)

    os.makedirs(WINRATE_SAVE_DIR, exist_ok=True)
    answer_winrate_df.to_csv(
        WINRATE_SAVE_DIR / "pairwise_answer_winrate.csv",
        index=False,
    )
    answer_winrate_summary_df.to_csv(
        WINRATE_SAVE_DIR / "answer_winrate_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
