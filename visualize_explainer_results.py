#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent
EVALUATIONS_DIR = REPO_ROOT / "evaluations"


def parse_args() -> argparse.Namespace:
    evaluation_choices = sorted(
        path.name for path in EVALUATIONS_DIR.iterdir() if path.is_dir()
    )
    parser = argparse.ArgumentParser(
        description=(
            "Render explainer RESULTS.jsonl entries into readable text files, "
            "one file per answer."
        )
    )
    parser.add_argument(
        "--evaluation",
        choices=evaluation_choices,
        default="chatbs-base",
        help="Evaluation folder under evaluations/ to load results from.",
    )
    parser.add_argument(
        "--variant",
        default="results",
        help=(
            "Explainer variant folder under evaluations/<evaluation>/explainer/. "
            "Examples: results, fullcontext, llmbased, grasp."
        ),
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help=(
            "Experiment folder name under the selected variant. If omitted, the "
            "latest experiment containing RESULTS.jsonl is used."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Exact path to a RESULTS.jsonl file. If provided, it overrides "
            "--evaluation/--variant/--experiment."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write the text files into. Defaults to "
            "evaluations/<evaluation>/text_results/<variant>/<experiment_name>/"
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
    return rows


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input:
        return Path(args.input)

    explainer_dir = EVALUATIONS_DIR / args.evaluation / "explainer" / args.variant
    if not explainer_dir.exists():
        raise FileNotFoundError(
            f"Explainer variant directory not found: {explainer_dir}"
        )

    if args.experiment:
        input_path = explainer_dir / args.experiment / "RESULTS.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return input_path

    candidate_files = sorted(
        explainer_dir.glob("*/RESULTS.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidate_files:
        raise FileNotFoundError(
            "No RESULTS.jsonl files found under "
            f"{explainer_dir}. Pass --input or --experiment explicitly."
        )
    return candidate_files[0]


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "result"


def pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def is_root_entry_id(value: str) -> bool:
    return bool(value) and not value.isdigit()


def display_node_id(value: str) -> str:
    return "0" if is_root_entry_id(str(value)) else str(value)


def collect_nodes(root: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}

    def visit(node: Dict[str, Any]) -> None:
        node_id = str(node.get("id", "")).strip()
        if node_id:
            nodes[node_id] = node

        predecessors = node.get("predecessor_info", {})
        if not isinstance(predecessors, dict):
            return

        for predecessor in predecessors.values():
            if isinstance(predecessor, dict):
                visit(predecessor)

    visit(root)
    return nodes


def build_edges(nodes: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
    edges: set[Tuple[str, str]] = set()
    for node_id, node in nodes.items():
        predecessors = node.get("predecessor_info", {})
        if not isinstance(predecessors, dict):
            continue
        for predecessor_id in predecessors:
            edges.add((display_node_id(str(predecessor_id)), display_node_id(node_id)))
    return sorted(
        edges,
        key=lambda pair: (
            0 if pair[0].isdigit() else 1,
            int(pair[0]) if pair[0].isdigit() else 0,
            pair[0],
            0 if pair[1].isdigit() else 1,
            int(pair[1]) if pair[1].isdigit() else 0,
            pair[1],
        ),
    )


def sort_node_ids(node_ids: List[str]) -> List[str]:
    def sort_key(value: str) -> Tuple[int, int, str]:
        if str(value).isdigit():
            return (0, int(value), str(value))
        return (1, 0, str(value))

    return sorted(node_ids, key=sort_key)


def format_topology_graph(nodes: Dict[str, Dict[str, Any]]) -> str:
    edges = build_edges(nodes)
    lines: List[str] = []

    lines.append("Nodes:")
    for node_id in sort_node_ids(list(nodes.keys())):
        node = nodes[node_id]
        lines.append(
            f"[{display_node_id(node_id)}] {str(node.get('question', '')).strip()}"
        )

    lines.append("")
    lines.append("Edges:")
    if not edges:
        lines.append("(none)")
    else:
        for source_id, target_id in edges:
            lines.append(f"{source_id} -> {target_id}")

    lines.append("")
    lines.append("Adjacency:")
    has_adjacency = False
    outgoing: Dict[str, List[str]] = {node_id: [] for node_id in nodes}
    for source_id, target_id in edges:
        outgoing[source_id].append(target_id)
    for node_id in sort_node_ids(list(nodes.keys())):
        children = outgoing.get(node_id, [])
        if children:
            has_adjacency = True
            lines.append(f"{node_id}: {', '.join(children)}")
    if not has_adjacency:
        lines.append("(none)")

    return "\n".join(lines)


def format_plan(plan: Any) -> str:
    if not isinstance(plan, list) or not plan:
        return "None"

    lines: List[str] = []
    for index, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            lines.append(f"{index}. {step}")
            continue
        step_id = str(step.get("step_id", "")).strip() or f"step{index}"
        sub_question = str(step.get("sub_question", "")).strip()
        program_id = str(step.get("program_id", "")).strip()
        lines.append(f"{index}. {step_id}")
        if sub_question:
            lines.append(f"   Sub question: {sub_question}")
        if program_id:
            lines.append(f"   Program: {program_id}")
        remaining = {
            key: value
            for key, value in step.items()
            if key not in {"step_id", "sub_question", "program_id"}
        }
        if remaining:
            lines.append("   Extra:")
            lines.append(indent(pretty_json(remaining), "   "))
    return "\n".join(lines)


def format_intermediary_results(results: Any) -> str:
    if not isinstance(results, list) or not results:
        return "None"

    blocks: List[str] = []
    for index, step in enumerate(results, start=1):
        if not isinstance(step, dict):
            blocks.append(f"Step {index}:\n{step}")
            continue

        step_id = str(step.get("step_id", "")).strip() or f"step{index}"
        sub_question = str(step.get("sub_question", "")).strip()
        program_id = str(step.get("program_id", "")).strip()
        answer = str(step.get("answer", "")).strip()
        important_entities = step.get("important_entities", [])
        results_payload = step.get("results", {})

        lines = [f"Step {index}: {step_id}"]
        if sub_question:
            lines.append(f"Sub question: {sub_question}")
        if program_id:
            lines.append(f"Program: {program_id}")
        if answer:
            lines.append("Answer:")
            lines.append(answer)
        if important_entities:
            lines.append("Important entities:")
            if isinstance(important_entities, list):
                lines.extend(f"- {entity}" for entity in important_entities)
            else:
                lines.append(str(important_entities))
        if results_payload:
            lines.append("Results:")
            lines.append(pretty_json(results_payload))

        other_fields = {
            key: value
            for key, value in step.items()
            if key
            not in {
                "step_id",
                "sub_question",
                "program_id",
                "answer",
                "important_entities",
                "results",
            }
        }
        if other_fields:
            lines.append("Extra:")
            lines.append(pretty_json(other_fields))

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def indent(text: str, prefix: str) -> str:
    return "\n".join(
        f"{prefix}{line}" if line else prefix.rstrip()
        for line in text.splitlines()
    )


def format_judge(judge: Any) -> str:
    if not isinstance(judge, list) or not judge:
        return "None"
    return pretty_json(judge)


def format_sub_question_details(node: Dict[str, Any]) -> str:
    node_id = str(node.get("id", "")).strip()
    predecessors = node.get("predecessor_info", {})
    predecessor_ids = []
    if isinstance(predecessors, dict):
        predecessor_ids = sort_node_ids([str(key) for key in predecessors.keys()])

    lines = [
        f"Sub Question [{display_node_id(node_id)}]",
        f"Question: {str(node.get('question', '')).strip()}",
        f"Depends on: {', '.join(predecessor_ids) if predecessor_ids else 'None'}",
        "",
        "Grounding:",
        pretty_json(node.get("grounding", {})),
        "",
        "Synthetic Question Plan:",
        format_plan(node.get("synthetic_questions_plan")),
        "",
        "Intermediary Results:",
        format_intermediary_results(node.get("intermediary_results")),
        "",
        "Predecessor Context:",
        str(node.get("predecessor_context", "")).strip() or "None",
        "",
        "Step Context:",
        str(node.get("step_context", "")).strip() or "None",
        "",
        "Schema Reasoning:",
        pretty_json(node.get("schema_reasoning", {})),
        "",
        "Validation / Judge:",
        format_judge(node.get("judge")),
        "",
        "Answer:",
        str(node.get("answer", "")).strip() or "None",
    ]
    return "\n".join(lines).strip()


def render_entry(entry: Dict[str, Any]) -> str:
    nodes = collect_nodes(entry)
    ordered_ids = sort_node_ids(
        [
            node_id
            for node_id in nodes
            if node_id.isdigit() and node_id != "0"
        ]
    )
    sub_question_nodes = [nodes[node_id] for node_id in ordered_ids]

    sections = [
        "Original Question",
        str(entry.get("question", "")).strip(),
        "",
        "Sub Questions",
        "\n".join(
            f"[{display_node_id(str(node['id']))}] {str(node.get('question', '')).strip()}"
            for node in sub_question_nodes
        )
        or "None",
        "",
        "Topology Graph",
        format_topology_graph(nodes),
        "",
        "Details Of Each Sub Question Solving",
        "\n\n" + "\n\n".join(
            format_sub_question_details(node) for node in sub_question_nodes
        )
        if sub_question_nodes
        else "None",
        "",
        "Final Report",
        str(entry.get("report", "")).strip()
        or str(entry.get("answer", "")).strip()
        or "None",
        "",
        "Final Answer",
        str(entry.get("answer", "")).strip() or "None",
        "",
        "Final Validation",
        format_judge(entry.get("judge")),
        "",
        "Timing",
        f"{entry.get('time_taken', 'Unknown')} seconds",
    ]

    return "\n".join(sections).strip() + "\n"


def resolve_output_dir(input_path: Path, explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir)

    experiment_name = input_path.parent.name
    parts = input_path.parts

    try:
        evaluations_index = parts.index("evaluations")
        evaluation_name = parts[evaluations_index + 1]
        variant_name = parts[evaluations_index + 3]
    except (ValueError, IndexError) as exc:
        raise ValueError(
            "Could not infer evaluation and variant from input path. "
            "Pass --output-dir explicitly."
        ) from exc

    return (
        EVALUATIONS_DIR
        / evaluation_name
        / "text_results"
        / variant_name
        / experiment_name
    )


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = load_jsonl(input_path)
    output_dir = resolve_output_dir(input_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, entry in enumerate(rows):
        entry_id = str(entry.get("id", "")).strip() or f"row_{index:04d}"
        filename = sanitize_filename(entry_id) + ".txt"
        output_path = output_dir / filename
        output_path.write_text(render_entry(entry), encoding="utf-8")

    print(
        f"Wrote {len(rows)} text file(s) to {output_dir}"
    )


if __name__ == "__main__":
    main()
