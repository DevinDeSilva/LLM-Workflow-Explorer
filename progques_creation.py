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
        common_utils,
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

    ontology_triples_df = pd.read_csv(
        config.explorer_config.ontology_triples_path
        )
    ontology_triples_df.head()
    return


@app.cell
def _(Dict, ExecutableProgram, json):
    def path_to_question(exp:ExecutableProgram, exp1:ExecutableProgram, schema:Dict) -> str:
        exp1_q_path = exp1.metadata['path']
        exp1_ans = exp1.example_output.head(2)
        exp1_query = exp1.code
        exp1_question = "What was the program that generated this data item?"
    
        """
        Converts a single path (a list of node URIs) into a question-answering format utilizing an 
        LLM.
        """
        exp1 = """
        ### sparql query
        {exp_q}
        ### sparql result
        {exp_ans}
        ### path
        {path}
        ### question
        {question}
        """.format(
            path=exp1_q_path, 
            question=exp1_question,
            exp_q=exp1_query,
            exp_ans=exp1_ans.to_dict('list')
        )
    
        PROMPT_TEMPLATE = """
        Given the following SPARQL query and its result and the path it represents in the RDF graph, 
        generate a natural language question that would lead to this query and result.
    
        The symbolic definitions and an example are provided below.
        ##Definitions:
        {schema}
    
        ##Example1
        {exp1}
    
        ### sparql query
        {exp_q}
        ### sparql result
        {exp_ans}
        ### path
        {path}
        """
    
        return PROMPT_TEMPLATE.format(
            exp1=exp1, 
            schema=json.dumps(schema, indent=2),
            path=exp.metadata['path'], 
            exp_q=exp.code,
            exp_ans=exp.example_output.head(2).to_dict('list')
        )

    return


@app.cell
def _(common_utils, config):
    explorations = common_utils.serialization.load_pickle(
        config.explorer_config.exeprog_save_loc
        )
    return


if __name__ == "__main__":
    app.run()
