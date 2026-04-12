import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    import logging
    import pandas as pd
    from icecream import ic
    from typing import Any, Dict
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
    CONFIG_PATH = "evaluations/calibration/config.yaml"
    logging.info(f"Loading config: {CONFIG_PATH}")
    lconfig = load_config(CONFIG_PATH)
    config = ExperimentConfig.model_validate(lconfig)
    config
    return (
        Explainer,
        GTInfo,
        common_utils,
        config,
        create_timestamp_id,
        ic,
        logging,
        os,
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
    )

    os.makedirs(config.explainer_config.save_answer_loc, exist_ok=True)
    timestamp_exp = create_timestamp_id("exp_")

    return explainer, ground_truth, timestamp_exp


@app.cell
def _(common_utils, config, explainer, ground_truth, os, timestamp_exp, tqdm):
    for qinfo in tqdm(ground_truth.gt_info):
        pred = explainer.request(qinfo.question)
        common_utils.serialization.save_jsonl_append(
            os.path.join(config.explainer_config.save_answer_loc, f"{timestamp_exp}.jsonl"),
            pred
        )
    return


if __name__ == "__main__":
    app.run()
