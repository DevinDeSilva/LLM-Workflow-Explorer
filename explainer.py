import marimo

__generated_with = "0.20.4"
app = marimo.App()


app._unparsable_cell(
    r"""
    import logging
    import pandas as pd
    from icecream import ic
    from typing import List, Optional, Dict
    import os
    import dspy
    from tqdm import tqdm
    import dycomutils as common_utils
    from dotenv import load_dotenv
    from src.utils.graph_manager import GraphManager
    from src.utils.utils import load_config
    from src.config.experiment import ExperimentConfig
    from src.explainer

    from src.question_creation.ontology_info_retriever import OntologyInfoRetriever
    from src.llm import LLM

    load_dotenv()
    CONFIG_PATH = "evaluations/calibration/config.yaml"
    logging.info(f"Loading config: {CONFIG_PATH}")
    lconfig = load_config(CONFIG_PATH)
    config = ExperimentConfig.model_validate(lconfig)
    config
    """,
    name="_"
)


@app.cell
def _(config, logging):
    logging.basicConfig(
        filename=config.question_creation_config.log_file,               # Log to this file
        filemode='a',                     # 'a' for append, 'w' to overwrite each time
        level=logging.INFO,               # Capture INFO and above
        format='%(asctime)s - %(levelname)s - %(message)s', # Custom format
        datefmt='%Y-%m-%d %H:%M:%S'       # Custom date format
    )
    logger = logging.getLogger(__name__)
    return


app._unparsable_cell(
    r"""
    explainer = 
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
