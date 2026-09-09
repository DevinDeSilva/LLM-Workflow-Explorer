"""Shared config and output helpers for single-judge evaluation runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def resolve_repo_path(repo_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def resolve_evaluation_config_path(
    repo_root: Path,
    evaluation_name: str,
    config_value: str | Path | None = None,
) -> Path:
    """Resolve a default, evaluation-local, or repository-relative config path."""
    evaluation_dir = repo_root / "evaluations" / evaluation_name
    if config_value is None:
        config_path = evaluation_dir / "config.evaluation.yaml"
    else:
        raw_path = Path(config_value).expanduser()
        if raw_path.is_absolute():
            config_path = raw_path
        elif raw_path.parent == Path("."):
            config_path = evaluation_dir / raw_path
        else:
            config_path = repo_root / raw_path

    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")
    return config_path


def evaluation_judge_id(settings: Mapping[str, Any]) -> str | None:
    """Return the judge id exactly as declared by the selected config."""
    value = settings.get("judge_id")
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def resolve_configured_analysis_dir(
    repo_root: Path,
    settings: Mapping[str, Any],
    evaluation_name: str,
) -> Path:
    """Resolve the config's explicit output root without adding hidden folders."""
    judge_id = evaluation_judge_id(settings)
    if judge_id is not None and not settings.get("save_dir"):
        raise ValueError(
            f"Judge config {judge_id!r} must define evaluation.save_dir."
        )
    return resolve_repo_path(
        repo_root,
        settings.get(
            "save_dir",
            Path("evaluations") / evaluation_name / "analysis",
        ),
    )


def write_evaluation_manifest(
    output_dir: Path,
    values: Mapping[str, Any],
) -> Path:
    """Merge run metadata into a manifest without discarding another runner's data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest.update(loaded)
        except (json.JSONDecodeError, OSError):
            pass

    manifest.update(values)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path
