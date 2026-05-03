from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


EO_GENERATED_BY_CLASSES = {
    "workflow:Generative_Task",
    "workflow:Large_Language_Models",
    "workflow:Large_Language_Model_Output",
}


def unique_preserving_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def attribute_display(uri: str, attr: Dict[str, Any]) -> str:
    rdf_type = attr.get("rdf:type", {})
    if isinstance(rdf_type, dict):
        rdf_type = rdf_type.get("object", "-")

    entity_rep = ["{}[{}]".format(uri, rdf_type)]
    for relation, value in attr.items():
        if relation == "rdf:type":
            continue

        if not isinstance(value, dict):
            entity_rep.append("\t{} -> {}".format(relation, value))
            continue

        line = "\t{} -> {}".format(relation, value.get("object", ""))
        object_class = value.get("object_class", "-")
        line += "[{}]".format(object_class) if "-" != object_class else ""

        object_label = value.get("object_label", "-")
        if "-" != object_label:
            if len(object_label) > 20:
                label = object_label[:20] + " ..."
            else:
                label = object_label
            line += "({})".format(label)
        entity_rep.append(line)

    return "\n".join(entity_rep)


def _as_list(value: Any) -> List[Any]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    return [value]


def _format_important_entities(entities: Any) -> str:
    entities = _as_list(entities)
    if not entities:
        return ""

    cleaned = unique_preserving_order(
        [
            str(entity).strip()
            for entity in entities
            if entity is not None and str(entity).strip()
        ]
    )
    return ", ".join(cleaned)


def _step_is_fallback(step: Dict[str, Any]) -> bool:
    return (
        step.get("execution_mode") == "fallback-class-ttl"
        or step.get("strategy") == "class-ttl"
        or str(step.get("program_id", "")).startswith("fallback::")
    )


def _extract_step_entities(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (
        step.get("extracted_entities")
        or step.get("extracted_results")
        or []
    )


def _program_solves(
    synthetic_question_retriever: Any,
    program_id: Any,
) -> str:
    if synthetic_question_retriever is None:
        return ""

    try:
        program = synthetic_question_retriever.get_program_by_id(
            str(program_id or "")
        )
    except Exception:
        return ""

    if isinstance(program, dict):
        return str(program.get("solves", "") or "").strip()
    return str(program or "").strip()


def build_trace_report(
    record: Dict[str, Any],
    synthetic_question_retriever: Any = None,
    max_ent_per_step: int = 3,
) -> str:
    blocks = [
        "User Question:{}".format(record.get("question", "")),
        "Trace Answers",
    ]

    top_level_entities = _format_important_entities(
        _as_list(record.get("important_entities"))
        + _as_list(record.get("fallback_important_entities"))
    )
    if top_level_entities:
        blocks.append("Important Entities:\n{}".format(top_level_entities))

    intermediary_results = record.get("intermediary_results", [])
    has_fallback = bool(record.get("fallback")) or any(
        _step_is_fallback(step)
        for step in intermediary_results
        if isinstance(step, dict)
    )

    if not has_fallback:
        for step in intermediary_results:
            if not isinstance(step, dict):
                continue

            step_rep = [
                "Question: {}".format(step.get("sub_question", ""))
            ]

            if step.get("strategy") == "by_program":
                solves = _program_solves(
                    synthetic_question_retriever,
                    step.get("program_id", ""),
                )
                if solves:
                    step_rep.append("KG Question: {}".format(solves))
            elif step.get("strategy") == "by_linked_data":
                step_rep.append("Linked Entities:")

            step_rep.append(
                "Result Entities:\n{}".format(
                    _format_important_entities(step.get("important_entities", []))
                )
            )

            extracted_entities = _extract_step_entities(step)
            if extracted_entities:
                step_rep.append("Example Entities:")

                ent_types: Dict[str, Dict[str, int]] = {}
                for ent in extracted_entities:
                    attr_dict = {
                        attr["relation"]: attr
                        for attr in ent.get("attributes", [])
                        if isinstance(attr, dict) and "relation" in attr
                    }
                    ent_type = attr_dict.get("rdf:type", {}).get("object", "-")
                    if ent_type in ent_types:
                        ent_types[ent_type]["ent_count"] += 1
                        ent_types[ent_type]["attr_count"] = max(
                            ent_types[ent_type]["attr_count"],
                            len(attr_dict),
                        )
                    else:
                        ent_types[ent_type] = {
                            "ent_count": 1,
                            "attr_count": len(attr_dict),
                        }

                if len(ent_types) == 1:
                    entity_class = list(ent_types.keys())[0]
                    if ent_types[entity_class]["attr_count"] > 5:
                        ent = extracted_entities[0]
                        attr_dict = {
                            attr["relation"]: attr
                            for attr in ent.get("attributes", [])
                            if isinstance(attr, dict) and "relation" in attr
                        }
                        step_rep.append(
                            attribute_display(str(ent.get("uri", "")), attr_dict)
                        )
                    else:
                        for ent in extracted_entities[:max_ent_per_step]:
                            attr_dict = {
                                attr["relation"]: attr
                                for attr in ent.get("attributes", [])
                                if isinstance(attr, dict) and "relation" in attr
                            }
                            step_rep.append(
                                attribute_display(str(ent.get("uri", "")), attr_dict)
                            )
                else:
                    already_visited: set[str] = set()
                    for ent in extracted_entities:
                        attr_dict = {
                            attr["relation"]: attr
                            for attr in ent.get("attributes", [])
                            if isinstance(attr, dict) and "relation" in attr
                        }
                        ent_type = attr_dict.get("rdf:type", {}).get("object", "-")
                        if ent_type not in already_visited:
                            step_rep.append(
                                attribute_display(str(ent.get("uri", "")), attr_dict)
                            )

                        already_visited.add(ent_type)

            blocks.append("\n".join(step_rep))
    else:
        for step in intermediary_results:
            if not isinstance(step, dict):
                continue

            step_rep = [
                "Question: {}".format(
                    step.get("sub_question", record.get("question", ""))
                )
            ]

            if _step_is_fallback(step):
                step_rep.append("Fallback KG Context")
                fallback_classes = (
                    step.get("results", {}).get("fallback_classes", [])
                    if isinstance(step.get("results"), dict)
                    else []
                )
                if fallback_classes:
                    step_rep.append(
                        "Fallback Classes:\n{}".format(
                            _format_important_entities(fallback_classes)
                        )
                    )
                selection_reasoning = str(
                    step.get("selection_reasoning", "")
                ).strip()
                if selection_reasoning:
                    step_rep.append(
                        "Entity Selection Reasoning:\n{}".format(
                            selection_reasoning
                        )
                    )
            elif step.get("strategy") == "by_program":
                solves = _program_solves(
                    synthetic_question_retriever,
                    step.get("program_id", ""),
                )
                if solves:
                    step_rep.append("KG Question: {}".format(solves))
            elif step.get("strategy") == "by_linked_data":
                step_rep.append("Linked Entities:")

            step_rep.append(
                "Important Entities:\n{}".format(
                    _format_important_entities(step.get("important_entities", []))
                )
            )
            blocks.append("\n".join(step_rep))

    blocks.append("Summary Answer:\n{}".format(record.get("answer", "")))
    return "\n\n".join(blocks)


def _class_local_name(class_name: Any) -> str:
    class_name = str(class_name or "").strip()
    if not class_name:
        return ""
    if "#" in class_name:
        class_name = class_name.rsplit("#", 1)[-1]
    elif "/" in class_name:
        class_name = class_name.rstrip("/").rsplit("/", 1)[-1]
    if ":" in class_name:
        class_name = class_name.rsplit(":", 1)[-1]
    return class_name


def _class_matches(class_name: Any, target_classes: set[str]) -> bool:
    class_name = str(class_name or "").strip()
    if not class_name:
        return False

    target_locals = {_class_local_name(target) for target in target_classes}
    return class_name in target_classes or _class_local_name(class_name) in target_locals


def _uri_suggests_eo_generated_class(uri: Any) -> bool:
    local_name = _class_local_name(uri)
    return (
        local_name.startswith("Generative_Task")
        or local_name.startswith("LLM-")
        or local_name.startswith("LLM_Output")
    )


def _record_application_name(
    record: Dict[str, Any],
    application_name: Optional[str] = None,
) -> str:
    if application_name:
        return application_name

    for key in ("application_name", "app_name", "system_name"):
        value = str(record.get(key, "") or "").strip()
        if value:
            return value

    return "ChatBS"


def _extract_eo_generated_by_entities(record: Dict[str, Any]) -> List[str]:
    entities: List[str] = []

    for step in record.get("intermediary_results", []):
        if not isinstance(step, dict):
            continue

        for entity in _extract_step_entities(step):
            if not isinstance(entity, dict):
                continue

            attrs = entity.get("attributes", []) or []
            entity_uri = str(entity.get("uri", "") or "").strip()
            for attr in attrs:
                if not isinstance(attr, dict):
                    continue

                if attr.get("relation") == "rdf:type" and _class_matches(
                    attr.get("object"),
                    EO_GENERATED_BY_CLASSES,
                ):
                    entities.append(entity_uri)

                if _class_matches(attr.get("object_class"), EO_GENERATED_BY_CLASSES):
                    object_uri = str(attr.get("object", "") or "").strip()
                    if object_uri:
                        entities.append(object_uri)

        if _step_is_fallback(step):
            step_results = step.get("results", {})
            if isinstance(step_results, dict):
                object_uris = _as_list(step_results.get("object_uris"))
                entities.extend(
                    str(uri).strip()
                    for uri in object_uris
                    if _uri_suggests_eo_generated_class(uri)
                )
            entities.extend(_as_list(step.get("important_entities")))

    trace_context = str(record.get("step_context", "") or "")
    bracket_pattern = re.compile(
        r"([A-Za-z][\w-]*:[^\s\[\]]+)\[("
        + "|".join(re.escape(cls) for cls in EO_GENERATED_BY_CLASSES)
        + r")\]"
    )
    entities.extend(match.group(1) for match in bracket_pattern.finditer(trace_context))

    turtle_pattern = re.compile(
        r"([A-Za-z][\w-]*:[^\s;,.]+)\s+a\s+("
        + "|".join(re.escape(cls) for cls in EO_GENERATED_BY_CLASSES)
        + r")"
    )
    entities.extend(match.group(1) for match in turtle_pattern.finditer(trace_context))

    return unique_preserving_order(
        [
            str(entity).strip()
            for entity in entities
            if entity is not None and str(entity).strip()
        ]
    )


def _build_eo_system_trace(
    record: Dict[str, Any],
    synthetic_question_retriever: Any = None,
) -> str:
    step_blocks: List[str] = []
    intermediary_results = record.get("intermediary_results", [])
    fallback_steps = [
        step
        for step in intermediary_results
        if isinstance(step, dict) and _step_is_fallback(step)
    ]

    if bool(record.get("fallback")) or fallback_steps:
        fallback_entities: List[Any] = []
        fallback_entities.extend(_as_list(record.get("fallback_important_entities")))
        for step in fallback_steps:
            fallback_entities.extend(_as_list(step.get("important_entities")))
        if not fallback_entities:
            fallback_entities.extend(_as_list(record.get("important_entities")))

        return "\n".join(
            [
                "Question: {}".format(str(record.get("question", "")).strip()),
                "Fallback: these important entities were extracted from all KG entities loaded for the fallback classes.",
                "Important Entities:\n{}".format(
                    _format_important_entities(fallback_entities)
                ),
            ]
        )

    for step in intermediary_results:
        if not isinstance(step, dict):
            continue

        question = str(
            step.get("sub_question") or record.get("question", "")
        ).strip()
        step_rep = [f"Question: {question}"]

        if step.get("strategy") == "by_program":
            solves = _program_solves(
                synthetic_question_retriever,
                step.get("program_id", ""),
            )
            if solves:
                step_rep.append("KG Question: {}".format(solves))
        elif step.get("strategy") == "by_linked_data":
            step_rep.append("Linked Entities:")

        step_rep.append(
            "Important Entities:\n{}".format(
                _format_important_entities(step.get("important_entities", []))
            )
        )
        step_blocks.append("\n".join(step_rep))

    return "\n\n".join(step_blocks)


def build_eo_trace_report(
    record: Dict[str, Any],
    synthetic_question_retriever: Any = None,
    application_name: Optional[str] = None,
) -> str:
    system_name = _record_application_name(record, application_name)
    overall_answer = str(record.get("answer", "") or "").strip()
    generated_by = _format_important_entities(
        _extract_eo_generated_by_entities(record)
    )
    system_trace = _build_eo_system_trace(record, synthetic_question_retriever)

    return "\n\n".join(
        [
            "### Knowledge Based System:\n{}".format(system_name),
            (
                "### What were the system outputs associated with the user query "
                "and the system trace?:\n(System Recommendation)\n{}"
            ).format(overall_answer),
            (
                "### What are the entities associate with the question:\n"
                "(wasGeneratedBy)\n{}"
            ).format(generated_by),
            (
                "### Traces Associated with the system recommendation:\n"
                "(System Trace)\n{}"
            ).format(system_trace),
            "### Overall Answer to the Question:\n{}".format(overall_answer),
        ]
    )


def augment_ours_record_with_reports(
    record: Dict[str, Any],
    synthetic_question_retriever: Any = None,
    application_name: Optional[str] = None,
    answer_report: str = "original",
) -> Dict[str, Any]:
    augmented_record = dict(record)
    augmented_record["report"] = build_trace_report(
        augmented_record,
        synthetic_question_retriever=synthetic_question_retriever,
    )
    augmented_record["eo_report"] = build_eo_trace_report(
        augmented_record,
        synthetic_question_retriever=synthetic_question_retriever,
        application_name=application_name,
    )

    if answer_report == "eo":
        augmented_record["answer"] = augmented_record["eo_report"]
    elif answer_report == "trace":
        augmented_record["answer"] = augmented_record["report"]
    elif answer_report == "original":
        augmented_record["answer"] = record["answer"]
    else:
        ValueError("Wrong answer report value: {}".format(answer_report))
        
    return augmented_record
