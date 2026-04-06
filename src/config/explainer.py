from pydantic import BaseModel
from typing import Dict, Union

class ExplainerConfig(BaseModel):
    llm_type:str
    embedding_type:str
    vectordb_type:str
    save_answer_loc:str
    
    llm_config:Dict[str, Union[str, int, float]] = {}
    embedding_config:Dict[str, Union[str, int, float]] = {}
    vectordb_config:Dict[str, Union[str, int, float]] = {}
    object_search_config:Dict[str, Union[str, int, float]] = {}
    log_file: str