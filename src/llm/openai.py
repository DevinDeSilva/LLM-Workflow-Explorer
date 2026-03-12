from src.llm.base import BaseLlm
from src.config.llm.openai import OpenAILlmConfig
from icecream import ic

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install langchain_openai")

class OpenAILlm(BaseLlm):
    def __init__(self, config:OpenAILlmConfig):
        super().__init__(config)

    def _create_client(self):
        kwargs = {
            "openai_api_key": self.config.api_key,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "model_kwargs": {},
        }
        if self.config.top_p: # XXX: need to update
            kwargs["model_kwargs"]["top_p"] = self.config.top_p

        return ChatOpenAI(**kwargs)


    