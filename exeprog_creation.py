import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import logging
    import pandas as pd
    import os
    import dycomutils as common_utils
    from dotenv import load_dotenv
    from src.utils.graph_manager import GraphManager
    from src.utils.utils import load_config
    from src.config.experiment import ExperimentConfig

    logger = logging.getLogger(__name__)
    load_dotenv()
    CONFIG_PATH = "evaluations/chatbs/config.yaml"
    return (
        CONFIG_PATH,
        ExperimentConfig,
        GraphManager,
        load_config,
        logging,
        pd,
    )


@app.cell
def _(CONFIG_PATH, load_config, logging):
    logging.info(f"Loading config: {CONFIG_PATH}")
    lconfig = load_config(CONFIG_PATH)
    return (lconfig,)


@app.cell
def _(ExperimentConfig, lconfig):
    config = ExperimentConfig.model_validate(lconfig)
    config
    return (config,)


@app.cell
def _(GraphManager, config, pd):
    graph_manager = GraphManager(
            config.ttl, 
            config.file_paths.execution_kg_loc
        )

    ontology_triples_df = pd.read_csv(config.explorer_config.ontology_triples_path)
    return graph_manager, ontology_triples_df


@app.cell
def _(graph_manager):
    triples = graph_manager.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }", add_header_tail=False)
    triples.head(5)
    return


@app.cell
def _():
    return


@app.cell
def _(BFSExplorer, config, graph_manager, ontology_triples_df):
    workflow_explorer = BFSExplorer(
        kg_name="workflow", 
        graph_manager=graph_manager,
        ontology_info_triples=ontology_triples_df,
        parallel_execution=config.explorer_config.parallel,
        temp_folder=config.explorer_config.temp_folder
        )

    workflow_explorer.load_graph_and_schema(
        schema_fpath=config.file_paths.schema_loc,
        rdf_fpath=config.file_paths.execution_kg_loc,
        metadata_path=config.explorer_config.explorer_metadata_loc,
        use_cache=config.explorer_config.use_cache,
    )
    return (workflow_explorer,)


@app.cell
def _(config, workflow_explorer):
    workflow_explorer.explore_workflow_graph(
        save_loc=config.explorer_config.exeprog_save_loc
        )
    return


if __name__ == "__main__":
    app.run()
