from typing import Optional
from pydantic import BaseModel

class BaseLlmConfig(BaseModel):
    model:str
