from pydantic import BaseModel
from typing import List, Dict, Union

from src.config.base import BaseConfig
from src.config.explainer import ExplainerConfig
from src.config.question_creation import ExplorerConfig, QuestionCreationConfig

class InputFiles(BaseModel):
    schema_loc:str
    execution_kg_loc:str
    metadata_loc:str

class GTConfig(BaseModel):
    log_file: str
    save_loc: str

class TTLConfig(BaseModel):
    prefixes:List[Dict[str,str]]

class ExperimentConfig(BaseConfig):
    file_paths:InputFiles
    explorer_config:ExplorerConfig
    question_creation_config:QuestionCreationConfig
    explainer_config:ExplainerConfig  
    ttl:TTLConfig
    gt:GTConfig