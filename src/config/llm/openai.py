import os
from pydantic import Field, field_validator
from src.config.llm.base import BaseLLMConfig

class OpenAILlmConfig(BaseLLMConfig):
    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
        ) # type: ignore
    model: str = 'gpt-4o'
    temperature: float = 0.2  
    max_tokens: int = 8196
    top_p: float = 0.3

    @field_validator("model", mode="before")
    @classmethod
    def normalize_model(cls, value: str) -> str:
        if isinstance(value, str) and value.startswith("openai/"):
            return value.removeprefix("openai/")
        return value
