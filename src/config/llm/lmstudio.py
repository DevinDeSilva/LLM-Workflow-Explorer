import os
from pydantic import Field
from src.config.llm.base import BaseLLMConfig

class LMStudioConfig(BaseLLMConfig):
    model: str = 'openai/qwen3-30b-a3b-2507'  
    temperature: float = 0.7  
    max_tokens: int = 8192  
    top_p: float = 0.9
    base_url: str = Field(
        default_factory=lambda: os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
        )