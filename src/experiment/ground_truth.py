from pydantic import BaseModel
from typing import Optional, Dict, Union, List
import dycomutils as common_utils

class SPARQLTemplate(BaseModel):
    template: str
    description: str
    inputs: Optional[Dict[str, str]] = None

class GT(BaseModel):
    question: str
    answer: str
    entities: Union[List[Dict], List]
    sparql_querys: List[SPARQLTemplate]
    
class GTInfo:
    def __init__(self, path_to_file:str) -> None:
        self.path_to_file = path_to_file
        self.gt_info = self.load_data_from_file()
        
    def load_data_from_file(self) -> List[GT]: 
        gt_file = common_utils.serialization.load_json(self.path_to_file)
        gts = [GT(**v) for v in gt_file.values()]
        return gts