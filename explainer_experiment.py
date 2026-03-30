import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import logging
    import pandas as pd
    from icecream import ic
    from typing import List, Optional, Dict
    import os
    import dspy
    from tqdm import tqdm
    import dycomutils as common_utils
    from dotenv import load_dotenv
    from src.utils.utils import load_config
    from src.config.experiment import ExperimentConfig
    from src.explainer.explainer import Explainer
    from src.experiment.ground_truth import GTInfo

    load_dotenv()
    CONFIG_PATH = "evaluations/calibration/config.yaml"
    logging.info(f"Loading config: {CONFIG_PATH}")
    lconfig = load_config(CONFIG_PATH)
    config = ExperimentConfig.model_validate(lconfig)
    config
    return Explainer, GTInfo, config, logging, tqdm


@app.cell
def _(config, logging):
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
def _(Explainer, GTInfo, config):
    ground_truth = GTInfo(config.gt.save_loc)
    explainer = Explainer(config.explainer_config)
    return explainer, ground_truth


@app.cell
def _(explainer, ground_truth, tqdm):
    for qinfo in tqdm(ground_truth.gt_info):
        pred = explainer.request(qinfo.question)
        break
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
