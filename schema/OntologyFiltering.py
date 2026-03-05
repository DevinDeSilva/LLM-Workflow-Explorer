import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Ontology Filtering

    In this notebook we select the necessary classes for our modelling from the complete ontologies of

    1. [Explanation ontology](https://raw.githubusercontent.com/tetherless-world/explanation-ontology/master/Ontologies/v2/explanation-ontology.owl)
    2. [ProvOne ontology]()
    3. [Prov Ontology]()
    4. [SIO ontology](https://akswnc7.informatik.uni-leipzig.de/dstreitmatter/archivo/semanticscience.org/ontology--sio--owl/2020.10.24-025110/ontology--sio--owl_type=generatedDocu.html#d4e2465)
    """)
    return


@app.cell
def _():
    from altair.datasets import load
    import os
    import sys
    import pandas as pd
    from icecream import ic
    import dycomutils as common_utils  
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.utils.graph_manager import GraphManager
    from src.utils.utils import load_config

    # configs
    config = {
        "full_ontology_path": "WorkFlow.ttl",
        "extracted_ontology_triples_path": "extracted_ontology_triples.csv",
        "ontology_config_path": "ontology_config.yaml",
        "important_classes_path": "schemaV2.json"
    }

    config = common_utils.config.ConfigDict(config)
    onto_config = load_config(config.ontology_config_path)
    return GraphManager, common_utils, config, ic, onto_config, pd


@app.cell
def _(GraphManager, common_utils, config, onto_config):
    ontology_graph = GraphManager(onto_config, config.full_ontology_path)
    important_class_list = common_utils.serialization.load_json(config.important_classes_path)
    return important_class_list, ontology_graph


@app.cell
def _(ontology_graph):
    ontology_graph.config["namespaces"]
    return


@app.cell
def _(ic, ontology_graph, pd):
    def select_relevant_triples(
        triples: pd.DataFrame, 
        important_class_list: list,
        assertion: bool = True
        ) -> pd.DataFrame:
        triples = triples.loc[triples['s'].str.contains("http://") | triples['s'].str.contains("https://")]
        for col in triples.columns:
            triples[col] = triples[col].apply(lambda x: ontology_graph.reverse_curie(x) if isinstance(x, str) else x)
        triples = triples.loc[triples['s'].isin(important_class_list)]

        if assertion:
            assert len(set(important_class_list) - set(triples['s'].unique())) == 0, "Some important classes are missing in the triples"
        else:
            missing_classes = set(important_class_list) - set(triples['s'].unique())
            if len(missing_classes) > 0:
                ic(missing_classes)
        return triples

    return (select_relevant_triples,)


@app.cell
def _(ontology_graph):
    triples = ontology_graph.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }", add_header_tail=False)
    triples
    return (triples,)


@app.cell
def _(important_class_list, select_relevant_triples, triples):
    sel_class_triples = select_relevant_triples(triples, important_class_list["classes"])
    sel_class_triples
    return (sel_class_triples,)


@app.cell
def _(important_class_list, select_relevant_triples, triples):
    sel_rela_triples = select_relevant_triples(
        triples, 
        list(important_class_list["object_properties"].keys()),
        assertion=False
        )
    sel_rela_triples
    return (sel_rela_triples,)


@app.cell
def _(config, pd, sel_class_triples, sel_rela_triples):
    combined_triples = pd.concat([sel_class_triples, sel_rela_triples], ignore_index=True)
    combined_triples.to_csv(config.extracted_ontology_triples_path, index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
