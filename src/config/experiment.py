from pydantic import BaseModel
from typing import List, Dict

from src.config.base import BaseConfig

class InputFiles(BaseModel):
    schema_loc:str
    execution_kg_loc:str
    metadata_loc:str
    
class ExplorerConfig(BaseModel):
    kg_name: str
    ontology_triples_path: str
    parallel: bool = False
    temp_folder: str = "tmp/programs"
    use_cache: bool = False
    explorer_metadata_loc: str
    exeprog_save_loc: str


class QuestionCreationConfig(BaseModel):
    save_questions:str
    
class TTLConfig(BaseModel):
    prefixes:List[Dict[str,str]]

class ExperimentConfig(BaseConfig):
    file_paths:InputFiles
    explorer_config:ExplorerConfig
    question_creation_config:QuestionCreationConfig
    ttl:TTLConfig