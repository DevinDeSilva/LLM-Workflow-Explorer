import marimo

__generated_with = "0.23.2"
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
    import os
    import dycomutils as common_utils
    from icecream import ic
    from dotenv import load_dotenv
    from src.utils.graph_manager import GraphManager
    from src.utils.utils import load_config
    from src.config.experiment import ExperimentConfig
    from src.explorer.bfs_explorer import BFSExplorer

    load_dotenv()
    ic(f"Loading config: {CONFIG_PATH}")
    lconfig = load_config(CONFIG_PATH)
    config = ExperimentConfig.model_validate(lconfig)
    config
    return BFSExplorer, GraphManager, config, ic, logging, os, pd


@app.cell
def _(config, ic, logging, os):
    os.makedirs(os.path.dirname(config.explorer_config.log_file), exist_ok=True)
    ic(config.explorer_config.log_file)
    logging.basicConfig(
        filename=config.explorer_config.log_file,               # Log to this file
        filemode='a',                     # 'a' for append, 'w' to overwrite each time
        level=logging.INFO,               # Capture INFO and above
        format='%(asctime)s - %(levelname)s - %(message)s', # Custom format
        datefmt='%Y-%m-%d %H:%M:%S'       # Custom date format
    )

    logger = logging.getLogger(__name__)  
    return


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
def _(BFSExplorer, config, graph_manager, ontology_triples_df):
    workflow_explorer = BFSExplorer(
        kg_name="workflow", 
        graph_manager=graph_manager,
        ontology_info_triples=ontology_triples_df,
        parallel_execution=config.explorer_config.parallel,
        temp_folder=config.explorer_config.temp_folder,
        entity_length = config.explorer_config.entity_length
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
