from pydantic import BaseModel
from typing import Dict, Union

class ExplainerConfig(BaseModel):
    llm_type:str
    model:str
    llm_config:Dict[str, Union[str, int, float]] = {}
    log_file: str