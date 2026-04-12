import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    import sys
    import logging
    from pydantic import BaseModel
    import os
    from pathlib import Path
    from typing import Dict, List, Optional, Union
    import pandas as pd

    import dycomutils as common_utils

    two_up = Path(__file__).resolve().parent.parent.parent
    os.chdir(two_up)
    print(two_up)
    sys.path.append(str(two_up))

    from src.utils.graph_manager import GraphManager  # noqa: E402
    from src.utils.utils import load_config # noqa: E402
    from src.config.experiment import ExperimentConfig # noqa: E402
    from src.experiment.ground_truth import GT, SPARQLTemplate # noqa: E402

    CONFIG_PATH = "evaluations/calibration/config.yaml"
    logging.info(f"Loading config: {CONFIG_PATH}")
    lconfig = load_config(CONFIG_PATH)
    config = ExperimentConfig.model_validate(lconfig)
    config
    return GT, GraphManager, List, SPARQLTemplate, common_utils, config, os


@app.cell
def _(GT, GraphManager, List, config):
    graph_manager = GraphManager(
            config.ttl, 
            config.file_paths.execution_kg_loc
        )

    gt_list:List[GT] = []
    return graph_manager, gt_list


@app.cell
def _(graph_manager):
    # Question 1
    question1 = """
    How many unique "experiment execution" are there in this?
    """
    question_sparql1 = """
    SELECT (count(distinct ?ids) AS ?obj_count)
    WHERE {
         ?obj a provone:Execution ;
              dcterms:identifier ?ids .  
    }
    """

    entities1 = graph_manager.query(question_sparql1)
    entities1
    return entities1, question1, question_sparql1


@app.cell
def _():
    answer1 = "The answer to the question is 1 unique executions."
    return (answer1,)


@app.cell
def _(
    GT,
    SPARQLTemplate,
    answer1,
    entities1,
    gt_list: "List[GT]",
    question1,
    question_sparql1,
):
    gt_list.append(
        GT(

            question=question1,
            answer=answer1,
            sparql_querys=[
                SPARQLTemplate(
                    template=question_sparql1,
                    description="This SPARQL query counts the number of unique executions by counting distinct identifiers in the provone:Execution class."
                )
            ],
            entities=entities1.to_dict(orient="records")
        )
    )
    return


@app.cell
def _(graph_manager):
    # Question 2

    question2 = """
    what are the instructions used by the LLM to generate the sparql query post processing function step in the pipeline?
    """
    question_sparql2 = """
    SELECT distinct ?obj ?llm ?inpd
    WHERE {
         ?obj dc:description ?desc .
         ?obj a ?class . 
         FILTER(REGEX(?desc, "Post-processes the query results","i"))

         ?llm_out sio:SIO_000202 ?obj .
         ?llm_out sio:SIO_000232 ?llm .
         ?llm sio:SIO_000230 ?inp .
         ?inp prov:value ?inpd
    }
    """

    entities2 = graph_manager.query(question_sparql2, resolve_curie=True)
    entities2
    return entities2, question2, question_sparql2


@app.cell
def _():
    answer2 = """
    We utilize the following input instructions to generate the function that is used to post process the sparql query results from the previous step.

    1. Concat all the rows into single row at each column by '|' concatenation marker.@en^^<xsd:string>
    2. If there are duplicate ingredients then select row with most information on `sugarG` and `ingredientCat` columns. 
       Fill all the missing values with '0' in `sugarG` column and with '-' in 'ingredientCat` column.@en^^<xsd:string>
    """
    return (answer2,)


@app.cell
def _(
    GT,
    SPARQLTemplate,
    answer2,
    entities2,
    gt_list: "List[GT]",
    question2,
    question_sparql2,
):
    gt_list.append(
        GT(
            question=question2,
            answer=answer2,
            sparql_querys=[
                SPARQLTemplate(
                    template=question_sparql2,
                    description=""
                )
            ],
            entities=entities2.to_dict(orient="records")
        )
    )
    return


@app.cell
def _(graph_manager):
    # Question 3  

    question3 = """
    In what places do we utilize AI in this workflow?
    """
    question_sparql3 = """
    SELECT distinct ?obj ?desc
    WHERE {
         ?obj a workflow:Generative_Task .
         ?obj dc:description ?desc .
    }
    """

    entities3 = graph_manager.query(question_sparql3, resolve_curie=True)
    entities3 
    return entities3, question3, question_sparql3


@app.cell
def _():
    answer3 = """
    The ChatBS System utilizes LLMs to generate the following functions

    1. Information extraction function
    2. System Prompt template function
    3. Sparql result post processing function

    and further it's used within functions to 
    1. Generate the LLM outputs,
    2. Used to extract key information from the system.
    """
    return (answer3,)


@app.cell
def _(
    GT,
    SPARQLTemplate,
    answer3,
    entities3,
    gt_list: "List[GT]",
    question3,
    question_sparql3,
):
    gt_list.append(
        GT(
            question=question3,
            answer=answer3,
            sparql_querys=[
                SPARQLTemplate(
                    template=question_sparql3,
                    description=""
                )
            ],
            entities=entities3.to_dict(orient="records")
        )
    )
    return


@app.cell
def _(graph_manager):

    # Question 4


    question4 = """
    What are the extracted KG entities for the execution with id 1_1 
    """
    question_sparql4 = """
    SELECT DISTINCT ?member ?prop ?value
    WHERE {
         ?obj dc:description ?desc .
         FILTER(REGEX(?desc, "SPARQL queries to extract information", "i"))

         ?obj a provone:Program .
         ?exe prov:qualifiedAssociation/prov:hadPlan ?obj .
         ?exe dcterms:identifier ?id . 
         FILTER(REGEX(?id, "1_1", "i"))

         ?data prov:wasGeneratedBy ?exe .
         ?data a ?class .

         {
             # Case 1: data is a Collection → get members and their properties
             ?data a provone:Collection .
             ?data provone:hadMember ?member .
             ?member ?prop ?value .
         }
         UNION
         {
             # Case 2: data is Data (not Collection) → get its properties
             ?data a provone:Data .
             BIND(?data AS ?member)
             ?data ?prop ?value .
         }
    }
    """


    entities4 = graph_manager.query(question_sparql4, resolve_curie=True)
    entities4
    return entities4, question4, question_sparql4


@app.cell
def _(entities4):
    pentities4 = set(entities4.loc[entities4["prop"] == "DFColumn:ingredientName", "value"].apply(lambda x: x.split("@")[0]).tolist()) - {'', 'NA'}
    return (pentities4,)


@app.cell
def _():
    answer4 = """
    The retrieved entities from the knowledge graph includes the following ingredients,

    1. Yogurt
    2. Berry
    3. Cheese
    4. Carrot
    5. Turkey
    6. Bell pepper
    7. Hummus
    8. Almond
    9. Avocado
    10. Pumpkin seed
    11. Water
    12. Pineapple
    13. Cottage cheese
    """
    return (answer4,)


@app.cell
def _(
    GT,
    SPARQLTemplate,
    answer4,
    gt_list: "List[GT]",
    pentities4,
    question4,
    question_sparql4,
):
    gt_list.append(
        GT(
            question=question4,
            answer=answer4,
            sparql_querys=[
                SPARQLTemplate(
                    template=question_sparql4,
                    description=""
                )
            ],
            entities=list(pentities4)
        )
    )
    return


@app.cell
def _(graph_manager):
    # Question 5

    question5 = """
    why did some ingredients have missing values for `sugarG` and `ingredientCat` columns in the final output of the workflow?
    """
    question_sparql5 = """
    SELECT DISTINCT ?member ?prop ?value
    WHERE {
         ?obj dc:description ?desc .
         FILTER(REGEX(?desc, "SPARQL queries to extract information", "i"))

         ?obj a provone:Program .
         ?exe prov:qualifiedAssociation/prov:hadPlan ?obj .
         ?exe dcterms:identifier ?id . 
         FILTER(REGEX(?id, "1_1", "i"))

         ?data prov:wasGeneratedBy ?exe .
         ?data a ?class .

         {
             # Case 1: data is a Collection → get members and their properties
             ?data a provone:Collection .
             ?data provone:hadMember ?member .
             ?member ?prop ?value .
         }
         UNION
         {
             # Case 2: data is Data (not Collection) → get its properties
             ?data a provone:Data .
             BIND(?data AS ?member)
             ?data ?prop ?value .
         }
    }

    order by ?member ?prop
    """

    entities5 = graph_manager.query(question_sparql5, resolve_curie=True)
    entities5
    return entities5, question5, question_sparql5


@app.cell
def _(entities5):
    na_entities= entities5.loc[entities5["value"].isin(['@en^^<xsd:string>', 'NA@en^^<xsd:string>']), "member"].unique().tolist()
    pentities5 = set(entities5.loc[entities5["member"].isin(na_entities) & (entities5["prop"] == "DFColumn:ingredientName"), "value"].apply(lambda x: x.split("@")[0]).tolist()) - {'', 'NA'}
    return (pentities5,)


@app.cell
def _():
    answer5 = """
    Due to the fact that some information is not available in the knowledge graph for ingredients such as 
    'Turkey', 'Hummus', 'Cheese', 'Cottage cheese', 'Berry', 'Water' the sugarG and ingredientCat is not available.
    """
    return (answer5,)


@app.cell
def _(
    GT,
    SPARQLTemplate,
    answer5,
    gt_list: "List[GT]",
    pentities5,
    question5,
    question_sparql5,
):
    gt_list.append(
        GT(
            question=question5,
            answer=answer5,
            sparql_querys=[
                SPARQLTemplate(
                    template=question_sparql5,
                    description=""
                )
            ],
            entities=list(pentities5)
        )
    )
    return


@app.cell
def _(common_utils, config, gt_list: "List[GT]", os):
    os.makedirs(os.path.dirname(config.gt.save_loc), exist_ok=True)
    for i, gt in enumerate(gt_list):
        gt.id = f"gt_{i}"

    common_utils.serialization.save_json(
        {gt.id: gt.model_dump() for gt in gt_list}, config.gt.save_loc
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
