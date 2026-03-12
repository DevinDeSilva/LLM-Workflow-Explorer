import os
from pydantic import Field
from src.config.llm.base import BaseLlmConfig

class LMStudioConfig(BaseLlmConfig):
    model: str = 'llama-3.3-70b-instruct'  
    temperature: float = 0.7  
    max_tokens: int = 8192  
    top_p: float = 0.9
    base_url: str = Field(
        default_factory=lambda: os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
        )