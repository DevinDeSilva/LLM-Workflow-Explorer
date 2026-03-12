import os
from pydantic import Field
from src.config.llm.base import BaseLlmConfig

class OpenAILlmConfig(BaseLlmConfig):
    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
        ) # type: ignore
    model: str = 'gpt-4-turbo'
    temperature: float = 0
    max_tokens: int = 1000
    top_p: float = 1