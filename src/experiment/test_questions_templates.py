from typing import Union
from collections.abc import Mapping, Sequence, Callable
from icecream import ic
import pandas as pd

from src.experiment.ground_truth import GT, SPARQLTemplate, GTAnswer
from src.utils.graph_manager import GraphManager
from src.utils.utils import regex_add_strings

AnswerTemplateFn = Callable[
    [pd.DataFrame, Mapping[str, Union[str, int]], Mapping[str, Union[str, int]]],
    GTAnswer
]

def build_gt_from_template(
    template:str, answer_template:AnswerTemplateFn, sparql_query:str, specs_template:Sequence[Mapping[str, Union[str,int]]], 
    specs_sparql:Sequence[Mapping[str, Union[str,int]]], graph_manager:GraphManager, verbose:bool):
    if len(specs_template) != len(specs_sparql):
        raise ValueError("specs_template and specs_sparql must have the same length")

    gt_instances: list[GT] = []

    for template_spec, sparql_spec in zip(specs_template, specs_sparql):
        query = regex_add_strings(sparql_query, **sparql_spec)
        entities = graph_manager.query(query)

        rquestion = regex_add_strings(template, **template_spec)               
        answer = answer_template(entities, template_spec, sparql_spec)
        
        if verbose:
            ic(rquestion)
            ic(entities)
            ic(answer)
        
        gt_instances.append(
            GT(
                question = rquestion,
                answer=answer.answer_nlp,
                entities=answer.entities,
                sparql=[SPARQLTemplate(template=query, description="")],
            )
        )
    return gt_instances

    
