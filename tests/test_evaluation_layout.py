from pathlib import Path

import pytest

from src.evaluation.layout import (
    evaluation_judge_id,
    resolve_configured_analysis_dir,
    resolve_evaluation_config_path,
)


def test_judge_config_uses_its_explicit_analysis_output(tmp_path: Path) -> None:
    settings = {
        "name": "sample",
        "save_dir": "evaluations/sample/analysis/judges/gpt-4o",
        "judge_id": "gpt-4o",
    }

    output_dir = resolve_configured_analysis_dir(tmp_path, settings, "sample")

    assert evaluation_judge_id(settings) == "gpt-4o"
    assert output_dir == (
        tmp_path
        / "evaluations"
        / "sample"
        / "analysis"
        / "judges"
        / "gpt-4o"
    )


def test_missing_judge_id_preserves_legacy_analysis_output(tmp_path: Path) -> None:
    settings = {
        "name": "sample",
        "save_dir": "evaluations/sample/analysis",
    }

    output_dir = resolve_configured_analysis_dir(tmp_path, settings, "sample")

    assert output_dir == tmp_path / "evaluations" / "sample" / "analysis"


def test_judge_config_requires_an_explicit_save_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must define evaluation.save_dir"):
        resolve_configured_analysis_dir(
            tmp_path,
            {"name": "sample", "judge_id": "qwen"},
            "sample",
        )


def test_config_basename_resolves_inside_evaluation_folder(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluations" / "sample" / "config.evaluation.qwen.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("evaluation: {}\n", encoding="utf-8")

    resolved = resolve_evaluation_config_path(
        tmp_path,
        "sample",
        "config.evaluation.qwen.yaml",
    )

    assert resolved == config_path


def test_missing_config_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Evaluation config not found"):
        resolve_evaluation_config_path(tmp_path, "sample", "missing.yaml")
