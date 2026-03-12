import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import logging
    import pandas as pd
    import os
    from dotenv import load_dotenv
    from src.explorer.bfs_explorer import BFSExplorer
    from src.utils.graph_manager import GraphManager
    from src.utils.utils import load_config

    logger = logging.getLogger(__name__)
    load_dotenv()
    CONFIG_PATH = "evaluations/chatbs/config.yaml"
    return CONFIG_PATH, GraphManager, load_config, logging, pd


@app.cell
def _(CONFIG_PATH, load_config, logging):
    logging.info(f"Loading config: {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    return (config,)


@app.cell
def _(config):
    config
    return


@app.cell
def _(GraphManager, config, pd):
    graph_manager = GraphManager(
            config, 
            config.file_paths.execution_kg_loc
        )

    ontology_triples_df = pd.read_csv(
        config.explorer_config.ontology_triples_path
        )
    ontology_triples_df.head()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
