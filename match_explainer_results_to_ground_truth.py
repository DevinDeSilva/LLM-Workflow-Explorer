#!/usr/bin/env python3
"""Match Biomni explainer result JSONL files to ground truth by question text."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_EXPLAINER_DIR = Path("evaluations/biomni-base/explainer")
DEFAULT_GROUND_TRUTH_PATH = Path(
    "evaluations/biomni-base/ground_truth/ground_truth_data.jsonl"
)
DEFAULT_EXCLUDED_DIRS = ("results",)
DEFAULT_OUTPUT_SUFFIX = ".matched_gt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create new explainer result JSONL files whose ground-truth fields "
            "are matched from ground_truth_data.jsonl by question text."
        )
    )
    parser.add_argument(
        "--explainer-dir",
        type=Path,
        default=DEFAULT_EXPLAINER_DIR,
        help=f"Directory to scan for result JSONL files. Default: {DEFAULT_EXPLAINER_DIR}",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_PATH,
        help=f"Ground-truth JSONL file. Default: {DEFAULT_GROUND_TRUTH_PATH}",
    )
    parser.add_argument(
        "--input-name",
        default="RESULTS.jsonl",
        help="Only process JSONL files with this name. Use '*.jsonl' to process all JSONL files.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=list(DEFAULT_EXCLUDED_DIRS),
        help=(
            "Directory name to exclude when scanning. Can be passed more than once. "
            "Default: results"
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default=DEFAULT_OUTPUT_SUFFIX,
        help=(
            "Suffix inserted before .jsonl for output files. "
            f"Default: {DEFAULT_OUTPUT_SUFFIX}"
        ),
    )
    parser.add_argument(
        "--keep-unmatched",
        action="store_true",
        help="Keep unmatched result rows in the output instead of writing only matched rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing matched output file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without creating files.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_question(question: Any) -> str:
    text = str(question or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_ground_truth_by_question(path: Path) -> dict[str, dict[str, Any]]:
    by_question: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        question_key = normalize_question(record.get("question"))
        if not question_key:
            raise ValueError(f"Ground-truth record has no question: {record.get('id')}")
        if question_key in by_question:
            existing_id = by_question[question_key].get("id")
            current_id = record.get("id")
            raise ValueError(
                "Duplicate ground-truth question after normalization: "
                f"{existing_id!r} and {current_id!r}"
            )
        by_question[question_key] = record
    return by_question


def first_user_message_index(record: dict[str, Any]) -> int | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return index
    return None


def extract_question(record: dict[str, Any]) -> tuple[str, str | None]:
    for key in ("question", "original_question"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value, key

    user_message_index = first_user_message_index(record)
    if user_message_index is not None:
        return record["messages"][user_message_index]["content"], "messages"

    return "", None


def replace_question(record: dict[str, Any], gt_question: str, source: str | None) -> None:
    if source in {"question", "original_question"}:
        record[source] = gt_question
    elif source == "messages":
        user_message_index = first_user_message_index(record)
        if user_message_index is not None:
            record["messages"][user_message_index]["content"] = gt_question

    if "question" in record:
        record["question"] = gt_question
    if "original_question" in record:
        record["original_question"] = gt_question.strip()


def replace_ground_truth_values(
    result_record: dict[str, Any],
    ground_truth_record: dict[str, Any],
    question_source: str | None,
) -> dict[str, Any]:
    updated = copy.deepcopy(result_record)
    matched_gt = copy.deepcopy(ground_truth_record)

    updated["id"] = matched_gt["id"]
    replace_question(updated, matched_gt["question"], question_source)

    if "gt" in updated:
        updated["gt"] = matched_gt
    updated["ground_truth"] = matched_gt
    for key, value in matched_gt.items():
        updated[f"ground_truth_{key}"] = copy.deepcopy(value)

    return updated


def iter_result_files(
    explainer_dir: Path,
    input_name: str,
    excluded_dirs: set[str],
    output_suffix: str,
) -> list[Path]:
    if input_name == "*.jsonl":
        candidates = explainer_dir.rglob("*.jsonl")
    else:
        candidates = explainer_dir.rglob(input_name)

    result_files: list[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        relative_parts = path.relative_to(explainer_dir).parts
        if any(part in excluded_dirs for part in relative_parts[:-1]):
            continue
        if path.stem.endswith(output_suffix):
            continue
        result_files.append(path)
    return sorted(result_files)


def output_path_for(input_path: Path, output_suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{output_suffix}{input_path.suffix}")


def process_file(
    path: Path,
    ground_truth_by_question: dict[str, dict[str, Any]],
    output_suffix: str,
    keep_unmatched: bool,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    output_path = output_path_for(path, output_suffix)
    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )

    matched_records: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for line_number, record in enumerate(read_jsonl(path), start=1):
        question, question_source = extract_question(record)
        gt = ground_truth_by_question.get(normalize_question(question))
        if gt is None:
            if keep_unmatched:
                unmatched_record = copy.deepcopy(record)
                unmatched_record["ground_truth"] = None
                for key in (
                    "id",
                    "question",
                    "answer",
                    "entities",
                    "sparql",
                    "qtype",
                    "decision",
                    "count",
                    "tags",
                ):
                    unmatched_record[f"ground_truth_{key}"] = None
                matched_records.append(unmatched_record)
            unmatched.append(
                {
                    "line": line_number,
                    "id": record.get("id"),
                    "question": question,
                }
            )
            continue

        matched_records.append(replace_ground_truth_values(record, gt, question_source))

    if not dry_run:
        write_jsonl(output_path, matched_records)

    return {
        "input": str(path),
        "output": str(output_path),
        "input_rows": len(matched_records) + (0 if keep_unmatched else len(unmatched)),
        "written_rows": len(matched_records),
        "matched_rows": len(matched_records) - (len(unmatched) if keep_unmatched else 0),
        "unmatched_rows": len(unmatched),
        "unmatched_examples": unmatched[:5],
    }


def main() -> None:
    args = parse_args()
    explainer_dir = args.explainer_dir.resolve()
    ground_truth_path = args.ground_truth.resolve()

    if not explainer_dir.exists():
        raise FileNotFoundError(f"Explainer directory not found: {explainer_dir}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {ground_truth_path}")

    ground_truth_by_question = load_ground_truth_by_question(ground_truth_path)
    result_files = iter_result_files(
        explainer_dir=explainer_dir,
        input_name=args.input_name,
        excluded_dirs=set(args.exclude_dir or []),
        output_suffix=args.output_suffix,
    )

    if not result_files:
        print(f"No matching result files found under {explainer_dir}")
        return

    print(f"Ground-truth rows loaded: {len(ground_truth_by_question)}")
    print(f"Result files to process: {len(result_files)}")

    summaries = [
        process_file(
            path=path,
            ground_truth_by_question=ground_truth_by_question,
            output_suffix=args.output_suffix,
            keep_unmatched=args.keep_unmatched,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        for path in result_files
    ]

    for summary in summaries:
        action = "would write" if args.dry_run else "wrote"
        print(
            f"{action} {summary['written_rows']} rows "
            f"({summary['matched_rows']} matched, {summary['unmatched_rows']} unmatched) "
            f"-> {summary['output']}"
        )
        for example in summary["unmatched_examples"]:
            question = str(example["question"] or "").strip().replace("\n", " ")
            print(
                f"  unmatched line {example['line']} id={example['id']!r}: "
                f"{question[:120]}"
            )


if __name__ == "__main__":
    main()
