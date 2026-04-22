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
        return str(Path("evaluations") / args.evaluation / "config.yaml")


    CONFIG_PATH = _parse_config_path()
    return (CONFIG_PATH,)


@app.cell
def _(CONFIG_PATH):
    import logging
    import pandas as pd
    from icecream import ic
    from typing import Any, Dict
    import time
    import os
    import dspy
    from tqdm import tqdm
    import dycomutils as common_utils
    from dotenv import load_dotenv
    from src.utils.utils import load_config
    from src.config.experiment import ExperimentConfig
    from src.explainer.explainer import Explainer
    from src.experiment.ground_truth import GTInfo
    from src.utils.utils import create_timestamp_id

    load_dotenv()
    logging.info(f"Loading config: {CONFIG_PATH}")
    lconfig = load_config(CONFIG_PATH)
    config = ExperimentConfig.model_validate(lconfig)
    return (
        Explainer,
        GTInfo,
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
        filename=config.explainer_config.log_file,               # Log to this file
        filemode='a',                     # 'a' for append, 'w' to overwrite each time
        level=logging.INFO,               # Capture INFO and above
        format='%(asctime)s - %(levelname)s - %(message)s', # Custom format
        datefmt='%Y-%m-%d %H:%M:%S'       # Custom date format
    )
    logger = logging.getLogger(__name__)
    return


@app.cell
def _(Explainer, GTInfo, config, create_timestamp_id, os):
    ground_truth = GTInfo(config.gt.save_loc)
    explainer = Explainer(
        config.file_paths.execution_kg_loc,
        config.file_paths.schema_loc,
        config.explorer_config.ontology_triples_path,
        config.explainer_config,
        config.application,
        config.ttl,
        config.question_creation_config.save_questions
    )

    os.makedirs(config.explainer_config.save_answer_loc, exist_ok=True)
    timestamp_exp = create_timestamp_id("exp_")
    os.makedirs(
        os.path.join(config.explainer_config.save_answer_loc, timestamp_exp),
        exist_ok=True
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
        report = explainer.request_to_report(pred)
        end_time = time.perf_counter()

        pred["question"] = qinfo.question
        pred["id"] = qinfo.id
        pred["report"] = report
        pred["time_taken"] = end_time - start_time
        common_utils.serialization.save_jsonl_append(
            os.path.join(
                config.explainer_config.save_answer_loc, timestamp_exp, "RESULTS.jsonl"
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
