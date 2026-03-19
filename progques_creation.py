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
    from src.utils.graph_manager import GraphManager
    from src.utils.utils import load_config
    from src.config.experiment import ExperimentConfig
    from src.explorer.executable_program import ExecutableProgram

    from src.question_creation.ontology_info_retriever import OntologyInfoRetriever
    from src.llm import LLM

    load_dotenv()
    CONFIG_PATH = "evaluations/chatbs/config.yaml"
    return (
        CONFIG_PATH,
        Dict,
        ExecutableProgram,
        ExperimentConfig,
        GraphManager,
        LLM,
        List,
        OntologyInfoRetriever,
        common_utils,
        dspy,
        ic,
        load_config,
        logging,
        pd,
        tqdm,
    )


@app.cell
def _(CONFIG_PATH, load_config, logging):
    logging.basicConfig(
        filename='evaluations/chatbs/ques_creation/exe.log',               # Log to this file
        filemode='a',                     # 'a' for append, 'w' to overwrite each time
        level=logging.INFO,               # Capture INFO and above
        format='%(asctime)s - %(levelname)s - %(message)s', # Custom format
        datefmt='%Y-%m-%d %H:%M:%S'       # Custom date format
    )
    logger = logging.getLogger(__name__)
    logging.info(f"Loading config: {CONFIG_PATH}")
    lconfig = load_config(CONFIG_PATH)
    return lconfig, logger


@app.cell
def _(ExperimentConfig, lconfig):
    config = ExperimentConfig.model_validate(lconfig)
    config
    return (config,)


@app.cell
def _(GraphManager, OntologyInfoRetriever, config):
    graph_manager = GraphManager(
            config.ttl, 
            config.file_paths.execution_kg_loc
        )

    ontology_triples = OntologyInfoRetriever(
        config.explorer_config.ontology_triples_path
        )

    ontology_triples.df.head()
    return (ontology_triples,)


@app.cell
def _(dspy, ic):
    from polars.selectors import by_index
    from sqlalchemy.sql.base import _exclusive_against
    class QuestionCreationFromPath(dspy.Signature):
        """
        Given a path in the ontology and SPARQL Query, what question is answered and information is retrived
        by the by utilizing this SPARQL query.

        The explanation should:
        - describe the entities being retrieved
        - describe filters or constraints
        - focus on semantic meaning
        """

        entity_info_triples: str = dspy.InputField(
            desc="entity details in the ontology"
            )

        path:str = dspy.InputField(
            desc="path in the ontology that is traversed"
            )

        sparql_query:str = dspy.InputField(
            desc="The sparql query responsible for retrieval"
            )

        question: str = dspy.OutputField(
            desc="Question answered by travelling throuugh that path"
            )

        information: str = dspy.OutputField(
            desc="Information Provided by travelling through that path"
            )

    class PathQuestionGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            # dspy.Predict uses the signature to generate the prompt structure
            self.generate_question = dspy.Predict(QuestionCreationFromPath)

        def forward(
            self, 
            entity_info_triples: str, 
            path: str, 
            sparql_query:str
        ) -> dspy.Prediction:
            """Executes the prediction logic."""
            ic(entity_info_triples,
                path,
                sparql_query)
            return self.generate_question(
                entity_info_triples=entity_info_triples,
                path=path,
                sparql_query=sparql_query
            )

    return (PathQuestionGenerator,)


@app.cell
def _(Dict, PathQuestionGenerator, dspy, ontology_triples):
    ex1:Dict[str,str] = {
        "path":"provone:Program->provone:hasInPort->provone:Port",
        "sparql_query":"""
        PREFIX provone: <http://purl.dataone.org/provone/2015/01/15/ontology#> 
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#> 
        PREFIX prov: <http://www.w3.org/ns/prov#> 
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
        PREFIX cwfo: <http://cwf.tw.rpi.edu/vocab#> 
        PREFIX dcterms: <http://purl.org/dc/terms#> 
        PREFIX ChatBS-NexGen: <http://testwebsite/testProgram#> 
        PREFIX user: <http://testwebsite/testUser#> 
        PREFIX eo: <https://purl.org/heals/eo#> 
        PREFIX ep: <http://linkedu.eu/dedalo/explanationPattern.owl#> 
        PREFIX sio: <http://semanticscience.org/resource/> 
        PREFIX dc: <http://purl.org/dc/elements/1.1/> 
        PREFIX DFColumn: <http://testwebsite/testDFColumn#> 
        PREFIX fnom: <https://w3id.org/function/vocabulary/mapping#> 
        PREFIX fnoi: <https://w3id.org/function/vocabulary/implementation#> 
        PREFIX fnoc: <https://w3id.org/function/vocabulary/composition/0.1.0/> 
        PREFIX food: <http://purl.org/heals/food/> 
        PREFIX dbo: <http://dbpedia.org/ontology/> 
        PREFIX dbp: <http://dbpedia.org/property/> 
        PREFIX dbt: <http://dbpedia.org/resource/Template:> 
        PREFIX ques: <http://atomic_questions.org/> 
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> 
        PREFIX fno: <https://w3id.org/function/vocabulary/core#> 
        PREFIX workflow: <http://www.semanticweb.org/acer/ontologies/2026/1/WorkFlow/> 

        SELECT distinct ?value where {
        <{obj}> provone:hasInPort  ?value  .
        } 
        """,
        "question": """
                    What are the inputs for the program/function specified?
                    """,
        "information":"""
                      For a given program/function the inputs variables are represented by this relationship.
                      """,
    }

    ex1_dspy:dspy.Example = dspy.Example(
            path=ex1["path"],
            entity_info_triples=ontology_triples.filter_triples(ex1["path"].split("->")),
            sparql_query = ex1["sparql_query"],
            question=ex1["question"],
            information=ex1["information"]
        ).with_inputs('path', 'entity_info_triples', 'sparql_query')

    trainset = [ex1_dspy]

    # 3. Initialize the optimizer
    # BootstrapFewShot will take our uncompiled module and our trainset,
    # and format the few-shot prompt for us under the hood.
    optimizer = dspy.teleprompt.BootstrapFewShot(metric=None) 

    # 4. Compile the module
    uncompiled_module = PathQuestionGenerator()
    compiled_module = optimizer.compile(uncompiled_module, trainset=trainset)
    return (compiled_module,)


@app.cell
def _(LLM, config):
    llm = LLM(
        config.question_creation_config.llm_type,
        "dspy",
        model = config.question_creation_config.model,
        **config.question_creation_config.llm_config
    )
    return


@app.cell
def _(ExecutableProgram, List, common_utils, config):
    explorations:List[ExecutableProgram] = common_utils.serialization.load_pickle(
        config.explorer_config.exeprog_save_loc
        )
    return (explorations,)


@app.cell
def _(
    compiled_module,
    explorations: "List[ExecutableProgram]",
    logger,
    ontology_triples,
    tqdm,
):
    exp_list = []

    for exp in tqdm(explorations):
        if 'path-level' in exp.tags:
            result = compiled_module(
                path='->'.join(exp.metadata['path']),
                entity_info_triples=ontology_triples.filter_triples(
                    exp.metadata['path']
                ),
                sparql_query = exp.code
            )

            logger.info(f"Generated question for path {exp.metadata['path']}: {result.question}")
            logger.debug(f"Generated information for path {exp.metadata['path']}: {result.information}\n")

            exp_dict = exp.to_dict()
            exp_dict.update({
                "category":'path-level',
                "solves": result.question,
                "statement": result.information,
                "start_node": exp.metadata["path"][0],
                "end_node": exp.metadata["path"][-1],
                "focal_relation":None,
                "focal_node": None
            })
            exp_list.append(
                exp_dict
            )



        elif 'object-level' in exp.tags and 'from-object' in exp.tags:
            _entity = list(set(exp.tags) - {'object-level', 'from-object'})[0]

            exp_dict = exp.to_dict()
            exp_dict.update({
                "category":'object-level|from-object',
                "solves": f"What is the value of {exp.metadata['relation']} of a object of class {_entity}?",
                "statement": f"Gives the {exp.metadata['relation']} attribute value of an object of class {_entity}",
                "start_node": None,
                "end_node": None,
                "focal_relation":exp.metadata['relation'],
                "focal_node": _entity
            })

            logger.info(f"Generated question for path {_entity} and relation {exp.metadata['relation']}: What is the value of {exp.metadata['relation']} of a object of class {_entity}?")

            exp_list.append(
                exp_dict
            )

        elif 'object-level' in exp.tags and 'from-prop' in exp.tags:
            _entity = list(set(exp.tags) - {'object-level', 'from-prop'})[0]

            exp_dict = exp.to_dict()
            exp_dict.update({
                "category":'object-level|from-prop',
                "solves": f"What are the objects of class {_entity} with the given value for {exp.metadata['relation']}?",
                "statement": f"Returns the objects attribute value of an object of class {_entity}",
                "start_node": None,
                "end_node": None,
                "focal_relation":exp.metadata['relation'],
                "focal_node": _entity
            })
            exp_list.append(
                exp_dict
            )

        elif 'class-level' in exp.tags:

            exp_dict = exp.to_dict()
            exp_dict.update({
                "category":'object-level|from-object',
                "solves": "What are all the objects of class?",
                "statement": "Returns all the objects of a given class",
                "start_node": None,
                "end_node": None,
                "focal_relation":None,
                "focal_node": None
            })
            exp_list.append(
                exp_dict
            )

        else:
            raise ValueError("Tags not in proper class:{}".format(exp.tags))
    return (exp_list,)


@app.cell
def _(config, exp_list, pd):
    exp_df = pd.DataFrame.from_records(exp_list)
    exp_df.to_csv(config.question_creation_config.save_questions, index=False)
    return


if __name__ == "__main__":
    app.run()
