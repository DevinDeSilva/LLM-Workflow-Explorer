from pydantic import BaseModel
from typing import Dict, Union

class ExplorerConfig(BaseModel):
    kg_name: str
    ontology_triples_path: str
    parallel: bool = False
    temp_folder: str = "tmp/programs"
    use_cache: bool = False
    explorer_metadata_loc: str
    exeprog_save_loc: str
    entity_length:int = 7
    log_file: str


class QuestionCreationConfig(BaseModel):
    save_questions:str
    llm_type:str
    llm_config:Dict[str, Union[str, int, float]] = {}
    log_file: str