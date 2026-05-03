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
            "Examples: results, fullcontext, llmbased, grasp, ground_truth. "
            "Use 'all' to render every variant under the selected evaluation."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Render every RESULTS.jsonl file under "
            "evaluations/<evaluation>/explainer/."
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
    parser.add_argument(
        "--ground-truth",
        default=None,
        help=(
            "Path to a ground_truth_data.jsonl file. Defaults to "
            "evaluations/<evaluation>/ground_truth/ground_truth_data.jsonl."
        ),
    )
    parser.add_argument(
        "--skip-ground-truth",
        action="store_true",
        help="When using --all, do not render the separate ground_truth text set.",
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


def infer_evaluation_dir(input_paths: List[Path], args: argparse.Namespace) -> Path:
    for input_path in input_paths:
        parts = input_path.parts
        try:
            evaluations_index = parts.index("evaluations")
            return Path(*parts[: evaluations_index + 2])
        except (ValueError, IndexError):
            continue
    return EVALUATIONS_DIR / args.evaluation


def resolve_ground_truth_path(input_paths: List[Path], args: argparse.Namespace) -> Path:
    if args.ground_truth:
        return Path(args.ground_truth)
    return (
        infer_evaluation_dir(input_paths, args)
        / "ground_truth"
        / "ground_truth_data.jsonl"
    )


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input:
        return Path(args.input)

    if args.variant == "ground_truth":
        raise ValueError("Use --variant ground_truth without resolving explainer input.")

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


def resolve_input_paths(args: argparse.Namespace) -> List[Path]:
    if args.variant == "ground_truth":
        if args.input:
            raise ValueError("--input cannot be combined with --variant ground_truth")
        return []

    if args.input:
        if args.all or args.variant == "all":
            raise ValueError("--input cannot be combined with --all or --variant all")
        return [Path(args.input)]

    if args.all or args.variant == "all":
        explainer_dir = EVALUATIONS_DIR / args.evaluation / "explainer"
        if not explainer_dir.exists():
            raise FileNotFoundError(f"Explainer directory not found: {explainer_dir}")

        candidate_files = sorted(explainer_dir.glob("*/*/RESULTS.jsonl"))
        if args.experiment:
            candidate_files = [
                path for path in candidate_files if path.parent.name == args.experiment
            ]
        if not candidate_files:
            raise FileNotFoundError(
                "No RESULTS.jsonl files found under "
                f"{explainer_dir}. Pass --input or check --evaluation."
            )
        return candidate_files

    return [resolve_input_path(args)]


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "result"


def pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return pretty_json(value)
    return str(value).strip()


def append_section(lines: List[str], title: str, value: Any) -> None:
    text = stringify(value)
    if not text:
        text = "None"
    lines.extend([title, text, ""])


def format_value(value: Any) -> str:
    text = stringify(value)
    return text if text else "None"


def format_label_id(label: Any, item_id: Any) -> str:
    label_text = stringify(label)
    id_text = stringify(item_id)
    if label_text and id_text and label_text != id_text:
        return f"{label_text} ({id_text})"
    return label_text or id_text or "None"


def render_ground_truth_entry(entry: Dict[str, Any]) -> str:
    shown_fields = {
        "id",
        "question",
        "answer",
        "qtype",
        "decision",
        "count",
        "entities",
        "sparql",
        "tags",
    }

    sections: List[str] = []
    append_section(sections, "Ground Truth ID", entry.get("id"))
    append_section(sections, "Question", entry.get("question"))
    append_section(sections, "Answer", entry.get("answer"))
    append_section(sections, "Question Type", entry.get("qtype"))
    append_section(sections, "Decision", entry.get("decision"))
    append_section(sections, "Count", entry.get("count"))
    append_section(sections, "Entities", entry.get("entities"))
    append_section(sections, "SPARQL", entry.get("sparql"))
    append_section(sections, "Tags", entry.get("tags"))

    additional_fields = {
        key: value for key, value in entry.items() if key not in shown_fields
    }
    if additional_fields:
        append_section(
            sections,
            "Additional Fields",
            pretty_json(additional_fields),
        )
    append_section(sections, "Raw Ground Truth", pretty_json(entry))
    return "\n".join(sections).strip() + "\n"


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
    outgoing: Dict[str, List[str]] = {
        display_node_id(node_id): [] for node_id in nodes
    }
    for source_id, target_id in edges:
        outgoing[source_id].append(target_id)
    for node_id in sort_node_ids(list(nodes.keys())):
        display_id = display_node_id(node_id)
        children = outgoing.get(display_id, [])
        if children:
            has_adjacency = True
            lines.append(f"{display_id}: {', '.join(children)}")
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
        step_id = str(step.get("step_id", "")).strip()
        if not step_id and "round" in step:
            step_id = f"round {step.get('round')}"
        step_id = step_id or f"step{index}"
        sub_question = str(
            step.get("sub_question", step.get("question", ""))
        ).strip()
        program_id = str(step.get("program_id", "")).strip()
        object_uris = step.get("object_uris")
        lines.append(f"{index}. {step_id}")
        if sub_question:
            lines.append(f"   Question: {sub_question}")
        if program_id:
            lines.append(f"   Program: {program_id}")
        if isinstance(object_uris, list):
            lines.append(f"   Object URIs ({len(object_uris)}):")
            lines.extend(f"   - {uri}" for uri in object_uris)
        remaining = {
            key: value
            for key, value in step.items()
            if key
            not in {
                "round",
                "step_id",
                "sub_question",
                "question",
                "program_id",
                "object_uris",
            }
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
    if not judge:
        return "None"
    return pretty_json(judge)


def format_sub_question_details(node: Dict[str, Any]) -> str:
    node_id = str(node.get("id", "")).strip()
    predecessors = node.get("predecessor_info", {})
    predecessor_ids = []
    if isinstance(predecessors, dict):
        predecessor_ids = sort_node_ids([str(key) for key in predecessors.keys()])

    shown_fields = {
        "id",
        "question",
        "predecessor_info",
        "grounding",
        "answer_requirements",
        "retrieved_objects",
        "schema_info_used",
        "synthetic_questions_plan",
        "intermediary_results",
        "predecessor_context",
        "step_context",
        "schema_reasoning",
        "judge",
        "answer",
        "report",
        "time_taken",
        "original_question",
    }
    additional_fields = {
        key: value for key, value in node.items() if key not in shown_fields
    }

    lines: List[str] = [
        f"Question [{display_node_id(node_id)}]",
        f"Question: {str(node.get('question', '')).strip()}",
        f"Depends on: {', '.join(predecessor_ids) if predecessor_ids else 'None'}",
        "",
        "Grounding:",
        pretty_json(node.get("grounding", {})),
        "",
        "Answer Requirements:",
        format_value(node.get("answer_requirements")),
        "",
        "Retrieved Objects:",
        format_value(node.get("retrieved_objects")),
        "",
        "Schema Info Used:",
        format_value(node.get("schema_info_used")),
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
    if additional_fields:
        lines.extend(["", "Additional Fields:", pretty_json(additional_fields)])
    return "\n".join(lines).strip()


def render_dependency_entry(entry: Dict[str, Any]) -> str:
    nodes = collect_nodes(entry)
    ordered_ids = sort_node_ids(
        [
            node_id
            for node_id in nodes
            if node_id.isdigit() and node_id != "0"
        ]
    )
    sub_question_nodes = [nodes[node_id] for node_id in ordered_ids]
    detail_nodes = sub_question_nodes if sub_question_nodes else [entry]

    original_question = str(
        entry.get("original_question", entry.get("question", ""))
    ).strip()
    current_question = str(entry.get("question", "")).strip()

    sections: List[str] = []
    append_section(sections, "Result ID", entry.get("id"))
    append_section(sections, "Original Question", original_question)
    if current_question and current_question != original_question:
        append_section(sections, "Resolved Question", current_question)
    append_section(
        sections,
        "Sub Questions",
        "\n".join(
            f"[{display_node_id(str(node['id']))}] "
            f"{str(node.get('question', '')).strip()}"
            for node in sub_question_nodes
        )
        or "None",
    )
    append_section(sections, "Topology Graph", format_topology_graph(nodes))
    append_section(
        sections,
        "Details Of Question Solving",
        "\n\n".join(format_sub_question_details(node) for node in detail_nodes),
    )
    append_section(
        sections,
        "Final Report",
        str(entry.get("report", "")).strip()
        or str(entry.get("answer", "")).strip()
        or "None",
    )
    append_section(sections, "Final Answer", entry.get("answer"))
    append_section(sections, "Final Validation", format_judge(entry.get("judge")))
    append_section(sections, "Timing", f"{entry.get('time_taken', 'Unknown')} seconds")

    return "\n".join(sections).strip() + "\n"


def format_entities(entities: Any) -> str:
    if not isinstance(entities, list) or not entities:
        return "None"

    lines: List[str] = []
    for index, entity in enumerate(entities, start=1):
        if not isinstance(entity, dict):
            lines.append(f"{index}. {entity}")
            continue
        name = format_label_id(entity.get("label"), entity.get("id"))
        meta = []
        if entity.get("types"):
            meta.append(f"types={entity.get('types')}")
        if entity.get("score") is not None:
            meta.append(f"score={entity.get('score')}")
        suffix = f" [{'; '.join(meta)}]" if meta else ""
        lines.append(f"{index}. {name}{suffix}")

    lines.extend(["", "Raw Relevant Entities:", pretty_json(entities)])
    return "\n".join(lines)


def format_evidence(evidence: Any) -> str:
    if not isinstance(evidence, list) or not evidence:
        return "None"

    lines: List[str] = []
    for index, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            lines.append(f"{index}. {item}")
            continue
        subject = format_label_id(item.get("subject_label"), item.get("subject_id"))
        predicate = format_label_id(
            item.get("predicate_label"), item.get("predicate_id")
        )
        object_value = format_label_id(item.get("object_label"), item.get("object_id"))
        meta = []
        if item.get("direction"):
            meta.append(f"direction={item.get('direction')}")
        if item.get("object_is_literal") is not None:
            meta.append(f"literal={item.get('object_is_literal')}")
        if item.get("score") is not None:
            meta.append(f"score={item.get('score')}")
        suffix = f" [{'; '.join(meta)}]" if meta else ""
        lines.append(f"{index}. {subject} -- {predicate} -> {object_value}{suffix}")

    lines.extend(["", "Raw Evidence:", pretty_json(evidence)])
    return "\n".join(lines)


def render_evidence_entry(entry: Dict[str, Any]) -> str:
    sections: List[str] = []
    append_section(sections, "Result ID", entry.get("id"))
    append_section(sections, "Question", entry.get("question"))
    append_section(sections, "Answer", entry.get("answer"))
    append_section(
        sections,
        "Relevant Entities",
        format_entities(entry.get("relevant_entities")),
    )
    append_section(sections, "Evidence", format_evidence(entry.get("evidence")))
    append_section(sections, "Timing", f"{entry.get('time_taken', 'Unknown')} seconds")

    remaining = {
        key: value
        for key, value in entry.items()
        if key
        not in {
            "id",
            "question",
            "answer",
            "relevant_entities",
            "evidence",
            "time_taken",
        }
    }
    if remaining:
        append_section(sections, "Additional Fields", pretty_json(remaining))
    return "\n".join(sections).strip() + "\n"


def extract_user_question(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = stringify(message.get("content"))
        if content:
            return content
    return ""


def format_grasp_output(output: Any) -> str:
    if output is None:
        return "None"
    if not isinstance(output, dict):
        return stringify(output)

    lines: List[str] = []
    append_section(lines, "Output Type", output.get("type"))
    if output.get("answer"):
        append_section(lines, "Answer", output.get("answer"))
    if output.get("explanation"):
        append_section(lines, "Explanation", output.get("explanation"))
    if output.get("formatted"):
        append_section(lines, "Formatted", output.get("formatted"))
    if output.get("sparql"):
        append_section(lines, "SPARQL", output.get("sparql"))
    if output.get("kg"):
        append_section(lines, "Knowledge Graph", output.get("kg"))
    if output.get("endpoint"):
        append_section(lines, "Endpoint", output.get("endpoint"))
    if output.get("selections"):
        append_section(lines, "Selections", output.get("selections"))
    if output.get("result"):
        append_section(lines, "Execution Result", output.get("result"))

    remaining = {
        key: value
        for key, value in output.items()
        if key
        not in {
            "type",
            "answer",
            "explanation",
            "formatted",
            "sparql",
            "kg",
            "endpoint",
            "selections",
            "result",
        }
    }
    if remaining:
        append_section(lines, "Additional Output Fields", pretty_json(remaining))
    return "\n".join(lines).strip()


def format_messages(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        return "None"

    blocks: List[str] = []
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            blocks.append(f"Message {index}:\n{message}")
            continue

        role = stringify(message.get("role")) or "unknown"
        lines = [f"Message {index}: {role}"]
        content = stringify(message.get("content"))
        if content:
            lines.extend(["Content:", content])
        extra_fields = {
            key: value
            for key, value in message.items()
            if key not in {"role", "content"}
        }
        if extra_fields:
            lines.extend(["Extra:", pretty_json(extra_fields)])
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def render_grasp_entry(entry: Dict[str, Any]) -> str:
    sections: List[str] = []
    append_section(sections, "Result ID", entry.get("id"))
    append_section(sections, "Task", entry.get("task"))
    append_section(sections, "Row Type", entry.get("type"))
    append_section(
        sections,
        "Question",
        extract_user_question(entry.get("messages")),
    )
    append_section(sections, "Error", entry.get("error"))
    append_section(sections, "Output", format_grasp_output(entry.get("output")))
    append_section(sections, "Known Items", format_value(entry.get("known")))
    append_section(sections, "Messages", format_messages(entry.get("messages")))
    append_section(sections, "Timing", f"{entry.get('time_taken', 'Unknown')} seconds")
    if entry.get("elapsed") is not None:
        append_section(sections, "Elapsed", f"{entry.get('elapsed')} seconds")

    remaining = {
        key: value
        for key, value in entry.items()
        if key
        not in {
            "id",
            "task",
            "type",
            "messages",
            "known",
            "output",
            "error",
            "elapsed",
            "time_taken",
        }
    }
    if remaining:
        append_section(sections, "Additional Fields", pretty_json(remaining))
    return "\n".join(sections).strip() + "\n"


def render_generic_entry(entry: Dict[str, Any]) -> str:
    sections: List[str] = []
    append_section(sections, "Result ID", entry.get("id"))
    if "question" in entry:
        append_section(sections, "Question", entry.get("question"))
    if "answer" in entry:
        append_section(sections, "Answer", entry.get("answer"))
    if "error" in entry:
        append_section(sections, "Error", entry.get("error"))
    if "time_taken" in entry:
        append_section(sections, "Timing", f"{entry.get('time_taken')} seconds")
    append_section(sections, "Raw Entry", pretty_json(entry))
    return "\n".join(sections).strip() + "\n"


def render_entry(entry: Dict[str, Any]) -> str:
    if any(
        key in entry
        for key in {
            "grounding",
            "predecessor_info",
            "synthetic_questions_plan",
            "intermediary_results",
            "schema_reasoning",
            "report",
            "judge",
        }
    ):
        return render_dependency_entry(entry)
    if "evidence" in entry or "relevant_entities" in entry:
        return render_evidence_entry(entry)
    if "messages" in entry or "output" in entry or "task" in entry:
        return render_grasp_entry(entry)
    return render_generic_entry(entry)


def resolve_output_dir(
    input_path: Path,
    explicit_output_dir: str | None,
    nest_explicit_output: bool = False,
) -> Path:
    experiment_name = input_path.parent.name

    if explicit_output_dir:
        output_dir = Path(explicit_output_dir)
        if not nest_explicit_output:
            return output_dir

        try:
            parts = input_path.parts
            evaluations_index = parts.index("evaluations")
            variant_name = parts[evaluations_index + 3]
        except (ValueError, IndexError):
            variant_name = input_path.parent.parent.name
        return output_dir / variant_name / experiment_name

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


def resolve_ground_truth_output_dir(
    ground_truth_path: Path,
    input_paths: List[Path],
    args: argparse.Namespace,
) -> Path:
    ground_truth_name = ground_truth_path.stem
    if args.output_dir:
        return Path(args.output_dir) / "ground_truth" / ground_truth_name

    evaluation_dir = infer_evaluation_dir(input_paths, args)
    return evaluation_dir / "text_results" / "ground_truth" / ground_truth_name


def write_text_rows(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    render,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(rows):
        entry_id = str(entry.get("id", "")).strip() or f"row_{index:04d}"
        filename = sanitize_filename(entry_id) + ".txt"
        output_path = output_dir / filename
        output_path.write_text(render(entry), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_paths = resolve_input_paths(args)

    total_rows = 0
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        rows = load_jsonl(input_path)
        output_dir = resolve_output_dir(
            input_path,
            args.output_dir,
            nest_explicit_output=len(input_paths) > 1,
        )
        write_text_rows(rows, output_dir, render_entry)
        total_rows += len(rows)
        print(f"Wrote {len(rows)} text file(s) to {output_dir}")

    if len(input_paths) > 1:
        print(f"Wrote {total_rows} text file(s) across {len(input_paths)} input file(s)")

    should_render_ground_truth = (
        args.variant == "ground_truth"
        or args.all
        or args.variant == "all"
    ) and not args.skip_ground_truth
    if should_render_ground_truth:
        ground_truth_path = resolve_ground_truth_path(input_paths, args)
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
        ground_truth_rows = load_jsonl(ground_truth_path)
        ground_truth_output_dir = resolve_ground_truth_output_dir(
            ground_truth_path,
            input_paths,
            args,
        )
        write_text_rows(
            ground_truth_rows,
            ground_truth_output_dir,
            render_ground_truth_entry,
        )
        print(
            f"Wrote {len(ground_truth_rows)} ground truth text file(s) to "
            f"{ground_truth_output_dir}"
        )


if __name__ == "__main__":
    main()
