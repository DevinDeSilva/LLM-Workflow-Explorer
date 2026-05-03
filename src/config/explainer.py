from pydantic import BaseModel
from typing import Dict, Union

class ExplainerConfig(BaseModel):
    llm_type:str
    embedding_type:str
    vectordb_type:str
    save_answer_loc:str
    
    llm_config:Dict[str, Union[str, int, float, bool]] = {}
    embedding_config:Dict[str, Union[str, int, float, bool]] = {}
    vectordb_config:Dict[str, Union[str, int, float, bool]] = {}
    object_search_config:Dict[str, Union[str, int, float, bool]] = {}
    log_file: str
