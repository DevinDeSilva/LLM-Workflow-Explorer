from pydantic import BaseModel, Field
from typing import Optional, Dict, Union, List
import uuid
import dycomutils as common_utils

class SPARQLTemplate(BaseModel):
    template: str
    description: str
    inputs: Optional[Dict[str, str]] = None

class GT(BaseModel):
    id:str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    answer: str
    entities: Union[List[Dict], List]
    sparql_querys: List[SPARQLTemplate]
    tags:Dict[str,List[str]] = Field(default={})

    def __str__(self) -> str:
        lines = [
            f"GT(id={self.id})",
            f"question: {self.question}",
            f"answer: {self.answer}",
            f"entities ({len(self.entities)}):",
        ]

        if self.entities:
            lines.extend(f"  - {entity}" for entity in self.entities)
        else:
            lines.append("  - None")

        lines.append(f"sparql_querys ({len(self.sparql_querys)}):")
        if self.sparql_querys:
            for sparql in self.sparql_querys:
                lines.append(f"  - template: {sparql.template}")
                lines.append(f"    description: {sparql.description}")
                lines.append(f"    inputs: {sparql.inputs if sparql.inputs is not None else {}}")
        else:
            lines.append("  - None")

        return "\n".join(lines)
    
class GTAnswer(BaseModel):
    answer_nlp:str
    entities: Union[List[Dict], List]
    
class GTInfo:
    def __init__(self, path_to_file:str) -> None:
        self.path_to_file = path_to_file
        self.gt_info = self.load_data_from_file()
        
    def load_data_from_file(self) -> List[GT]: 
        gt_file = common_utils.serialization.load_json(self.path_to_file)
        gts = [GT(**v) for v in gt_file.values()]
        return gts
