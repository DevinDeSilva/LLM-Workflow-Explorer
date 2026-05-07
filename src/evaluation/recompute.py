from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecomputeConfig:
    methods: set[str] | None
    question_ids: set[str] | None
    qtypes: set[str] | None
    reuse_existing: bool
    existing_results_dir: Path


def optional_str_set(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return {text} if text else set()
    return {str(item).strip() for item in value if str(item).strip()}


def merge_recompute_settings(
    base_settings: Mapping[str, Any] | None,
    override_settings: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(base_settings or {})
    for key, value in dict(override_settings or {}).items():
        if value is not None:
            merged[key] = value
    return merged


def build_recompute_config(
    base_settings: Mapping[str, Any] | None,
    override_settings: Mapping[str, Any] | None,
    default_existing_results_dir: Path,
    resolve_path: Callable[[str | Path], Path],
) -> RecomputeConfig:
    settings = merge_recompute_settings(base_settings, override_settings)
    existing_results_dir = settings.get("existing_results_dir") or settings.get("cache_dir")
    return RecomputeConfig(
        methods=optional_str_set(settings.get("methods")),
        question_ids=optional_str_set(
            settings.get("question_ids", settings.get("ground_truth_ids"))
        ),
        qtypes=optional_str_set(settings.get("qtypes")),
        reuse_existing=bool(settings.get("reuse_existing", True)),
        existing_results_dir=(
            resolve_path(existing_results_dir)
            if existing_results_dir
            else default_existing_results_dir
        ),
    )


def parse_qtypes(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        if text.startswith("["):
            try:
                return parse_qtypes(json.loads(text))
            except json.JSONDecodeError:
                pass
        return {text}
    return {str(item).strip() for item in value if str(item).strip()}


def qtypes_match(qtypes: Any, selected_qtypes: set[str] | None) -> bool:
    if selected_qtypes is None:
        return True
    return bool(parse_qtypes(qtypes) & selected_qtypes)


def validate_selected_methods(
    selected_methods: set[str] | None,
    available_methods: set[str],
    context: str,
) -> None:
    if selected_methods is None:
        return
    unknown_methods = selected_methods - available_methods
    if unknown_methods:
        raise ValueError(
            f"Unknown {context} recompute methods in config: "
            f"{sorted(unknown_methods)}. Available methods: {sorted(available_methods)}"
        )


def selected_ground_truth_ids(
    ground_truth_records: list[dict[str, Any]],
    recompute_config: RecomputeConfig,
) -> set[str] | None:
    if recompute_config.question_ids is None and recompute_config.qtypes is None:
        return None

    selected_ids: set[str] = set()
    for record in ground_truth_records:
        ground_truth_id = str(record.get("id") or "")
        if not ground_truth_id:
            continue
        if (
            recompute_config.question_ids is not None
            and ground_truth_id not in recompute_config.question_ids
        ):
            continue
        if not qtypes_match(record.get("qtype", []), recompute_config.qtypes):
            continue
        selected_ids.add(ground_truth_id)

    return selected_ids
