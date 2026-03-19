import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import logging
    import pandas as pd
    from src.explorer.bfs_explorer import BFSExplorer
    from src.utils.graph_manager import GraphManager
    from src.utils.utils import load_config

    logger = logging.getLogger(__name__)
    CONFIG_PATH = "evaluations/chatbs/exeprog_creation/config.yaml"
    return CONFIG_PATH, GraphManager, load_config, logging


@app.cell
def _(CONFIG_PATH, load_config, logging):
    logging.info(f"Loading config: {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    return (config,)


@app.cell
def _(GraphManager, config):
    graph_manager = GraphManager(
            config, 
            config.file_paths.execution_kg_loc
        )
    return (graph_manager,)


@app.cell
def _():
    SPARQL_QUERY = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX ep: <http://linkedu.eu/dedalo/explanationPattern.owl#>
    PREFIX eo: <https://purl.org/heals/eo#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    PREFIX food: <http://purl.org/heals/food/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX provone: <http://purl.dataone.org/provone/2015/01/15/ontology#>
    PREFIX sio:<http://semanticscience.org/resource/>

    SELECT distinct ?value where {
      ?value <http://purl.dataone.org/provone/2015/01/15/ontology#hadEntity>   <http://testwebsite/testProgram#Data-id_20260305222956_802-diagnosis> .
    }
    """
    return (SPARQL_QUERY,)


@app.cell
def _(SPARQL_QUERY, graph_manager):
    graph_manager.query(SPARQL_QUERY)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
