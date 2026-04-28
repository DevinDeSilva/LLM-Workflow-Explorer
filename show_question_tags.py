#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import shutil
import textwrap
from pathlib import Path
from typing import Iterable


DEFAULT_CSV = Path("evaluations/chatbs-base/ques_creation/SyntheticQuestionKG.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show question text and tag data side by side from a CSV file."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_CSV,
        type=Path,
        help=f"CSV file to read. Defaults to {DEFAULT_CSV}.",
    )
    parser.add_argument(
        "--question-column",
        default="solves",
        help="Column containing the question text.",
    )
    parser.add_argument(
        "--tags-column",
        default="tags",
        help="Column containing the tag data.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to print.",
    )
    parser.add_argument(
        "--raw-tags",
        action="store_true",
        help="Print tag data exactly as it appears in the CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Text file to save the table to. Defaults to "
            "<csv folder>/question_tags.txt."
        ),
    )
    parser.add_argument(
        "--print",
        dest="print_output",
        action="store_true",
        help="Also print the table to the terminal after saving it.",
    )
    return parser.parse_args()


def format_tags(value: str, raw: bool) -> str:
    if raw:
        return value

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value

    if isinstance(parsed, list):
        return ", ".join(str(item) for item in parsed)
    return str(parsed)


def wrap_cell(value: str, width: int) -> list[str]:
    lines: list[str] = []
    for part in str(value).splitlines() or [""]:
        wrapped = textwrap.wrap(
            part,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines


def table_rows(
    rows: Iterable[dict[str, str]],
    question_column: str,
    tags_column: str,
    raw_tags: bool,
    terminal_width: int,
) -> Iterable[str]:
    number_width = 5
    separator_width = 7
    available = max(60, terminal_width) - number_width - separator_width
    tags_width = min(42, max(24, available // 3))
    question_width = available - tags_width

    rule = f"{'-' * number_width}+{'-' * (question_width + 2)}+{'-' * (tags_width + 2)}"
    yield rule
    yield (
        f"{'#':>{number_width - 1}} "
        f"| {'Question':<{question_width}} "
        f"| {'Tags':<{tags_width}}"
    )
    yield rule

    for index, row in enumerate(rows, start=1):
        question = row.get(question_column, "")
        tags = format_tags(row.get(tags_column, ""), raw_tags)
        question_lines = wrap_cell(question, question_width)
        tag_lines = wrap_cell(tags, tags_width)
        height = max(len(question_lines), len(tag_lines))

        for line_index in range(height):
            row_number = str(index) if line_index == 0 else ""
            question_line = question_lines[line_index] if line_index < len(question_lines) else ""
            tag_line = tag_lines[line_index] if line_index < len(tag_lines) else ""
            yield (
                f"{row_number:>{number_width - 1}} "
                f"| {question_line:<{question_width}} "
                f"| {tag_line:<{tags_width}}"
            )
        yield rule


def main() -> None:
    args = parse_args()
    output_path = args.output or args.csv_path.with_name("question_tags.txt")

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {args.csv_path}")

        missing_columns = [
            column
            for column in (args.question_column, args.tags_column)
            if column not in reader.fieldnames
        ]
        if missing_columns:
            available = ", ".join(reader.fieldnames)
            missing = ", ".join(missing_columns)
            raise ValueError(f"Missing column(s): {missing}. Available columns: {available}")

        rows: Iterable[dict[str, str]] = reader
        if args.limit is not None:
            rows = (row for index, row in enumerate(reader) if index < args.limit)

        terminal_width = shutil.get_terminal_size((120, 20)).columns
        lines = list(
            table_rows(
                rows,
                question_column=args.question_column,
                tags_column=args.tags_column,
                raw_tags=args.raw_tags,
                terminal_width=terminal_width,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.print_output:
        for line in lines:
            print(line)

    print(f"Saved question/tag table to {output_path}")


if __name__ == "__main__":
    main()
