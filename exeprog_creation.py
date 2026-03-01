import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import logging
    from src.explorer.bfs_explorer import BFSExplorer
    from src.utils.graph_manager import GraphManager
    from src.utils.utils import load_config

    logger = logging.getLogger(__name__)
    CONFIG_PATH = "evaluations/chatbs/exeprog_creation/config.yaml"
    return BFSExplorer, CONFIG_PATH, GraphManager, load_config, logging


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
def _(GraphManager, config):
    graph_manager = GraphManager(
            config, 
            config.file_paths.execution_kg_loc
        )
    return (graph_manager,)


@app.cell
def _(BFSExplorer, config, graph_manager):
    workflow_explorer = BFSExplorer(kg_name="workflow", graph_manager=graph_manager)
    workflow_explorer.load_graph_and_schema(
        schema_fpath=config.file_paths.schema_loc,
        rdf_fpath=config.file_paths.execution_kg_loc,
        metadata_path=config.file_paths.explorer_metadata_loc,
        use_cache=True,
    )
    return (workflow_explorer,)


@app.cell
def _(config, workflow_explorer):
    workflow_explorer.explore_workflow_graph(
        save_loc=config.file_paths.exeprog_save_loc
        )
    return


if __name__ == "__main__":
    app.run()
