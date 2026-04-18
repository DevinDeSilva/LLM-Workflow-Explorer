import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import argparse
    from pathlib import Path
    import sys

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
            default="calibration-base",
            help="Evaluation folder under evaluations/ to load config from.",
        )
        args, remaining = parser.parse_known_args()
        sys.argv = [sys.argv[0], *remaining]
        return str(Path("evaluations") / args.evaluation / "config.llmbased.yaml")


    CONFIG_PATH = _parse_config_path()
    CONFIG_PATH
    return (CONFIG_PATH,)


@app.cell
def _(CONFIG_PATH):
    import logging
    import os
    import time

    import dycomutils as common_utils
    from dotenv import load_dotenv
    from icecream import ic
    from tqdm import tqdm

    from baselines.LLMbased import GroundedWorkflowBaseline
    from src.config.experiment import FullContextExperimentConfig
    from src.experiment.ground_truth import GTInfo
    from src.utils.utils import create_timestamp_id, load_config

    load_dotenv()
    logging.info(f"Loading config: {CONFIG_PATH}")
    lconfig = load_config(CONFIG_PATH)
    config = FullContextExperimentConfig.model_validate(lconfig)
    config
    return (
        GTInfo,
        GroundedWorkflowBaseline,
        common_utils,
        config,
        create_timestamp_id,
        ic,
        logging,
        os,
        time,
        tqdm,
    )


@app.cell
def _(config, ic, logging, os):
    os.makedirs(os.path.dirname(config.explainer_config.log_file), exist_ok=True)
    ic(config.explainer_config.log_file)

    logging.basicConfig(
        filename=config.explainer_config.log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    return


@app.cell
def _(GTInfo, GroundedWorkflowBaseline, config, create_timestamp_id, os):
    ground_truth = GTInfo(config.gt.save_loc)

    explainer = GroundedWorkflowBaseline(
        kg_path=config.file_paths.execution_kg_loc,
        ontology_path=config.file_paths.ontology_path,
        schema_json_path=config.file_paths.schema_loc,
        metadata_path=config.file_paths.metadata_loc,
        application_description=config.application.description,
        llm_type=config.explainer_config.llm_type,
        llm_library="langchain",
        llm_config=dict(config.explainer_config.llm_config),
    )

    timestamp_exp = create_timestamp_id("exp_")
    os.makedirs(
        os.path.join(config.explainer_config.save_answer_loc, timestamp_exp),
        exist_ok=True,
    )
    return explainer, ground_truth, timestamp_exp


@app.cell
def _(
    common_utils,
    config,
    explainer,
    ground_truth,
    os,
    time,
    timestamp_exp,
    tqdm,
):
    for qinfo in tqdm(ground_truth.gt_info):
        start_time = time.perf_counter()
        pred = explainer.request(qinfo.question)
        end_time = time.perf_counter()

        pred["time_taken"] = end_time - start_time
        common_utils.serialization.save_jsonl_append(
            os.path.join(
                config.explainer_config.save_answer_loc,
                timestamp_exp,
                "RESULTS.jsonl",
            ),
            pred,
        )

    common_utils.serialization.save_json(
        config.model_dump(),
        os.path.join(config.explainer_config.save_answer_loc, timestamp_exp, "config.json"),
    )
    return


if __name__ == "__main__":
    app.run()
