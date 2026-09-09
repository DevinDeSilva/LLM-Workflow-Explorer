from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_MODELS = {
    "qwen": ("qwen3.6-35b-a3b", "lmstudio", "qwen/qwen3.6-35b-a3b"),
    "chatgpt": ("gpt-5.4-mini", "openai", "gpt-5.4-mini"),
}


def test_single_judge_configs_have_isolated_outputs() -> None:
    for dataset in ("chatbs-base", "biomni-base"):
        output_dirs: set[str] = set()
        for config_name, (judge_id, llm_type, model) in EXPECTED_MODELS.items():
            config_path = (
                REPO_ROOT
                / "evaluations"
                / dataset
                / f"config.evaluation.{config_name}.yaml"
            )
            with config_path.open() as handle:
                settings = yaml.safe_load(handle)["evaluation"]

            assert settings["name"] == dataset
            assert settings["judge_id"] == judge_id
            assert settings["judge_llm"]["llm_type"] == llm_type
            assert settings["judge_llm"]["llm_config"]["model"] == model
            assert settings["winrate"]["judge_llm"]["llm_type"] == llm_type
            assert settings["winrate"]["judge_llm"]["llm_config"]["model"] == model
            assert settings["save_dir"].endswith(f"/analysis/judges/{judge_id}")
            output_dirs.add(settings["save_dir"])

        assert len(output_dirs) == len(EXPECTED_MODELS)
