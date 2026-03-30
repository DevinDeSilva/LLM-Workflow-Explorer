from pydantic import BaseModel
from typing import Dict, Union
from src.config.object_search import ObjectSearchConfig
from src.embeddings.base import 

class ExplainerConfig(BaseModel):
    llm_type:str
    embedding_type:str
    vectordb_type:str
    llm_config:Dict[str, Union[str, int, float]] = {}
    embedding_config:Dict[str, Union[str, int, float]] = {}
    vectordb_config:Dict[str, Union[str, int, float]] = {}
    object_search_config:ObjectSearchConfig = ObjectSearchConfig()
    log_file: str