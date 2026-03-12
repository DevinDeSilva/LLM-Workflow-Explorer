from src.llm.base import BaseLlm
from src.config.llm.lmstudio import LMStudioConfig
from icecream import ic


try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install langchain_openai")

class LMStudio(BaseLlm):
    def __init__(self, config:LMStudioConfig):
        super().__init__(config)
        

    def _create_client(self):
        
        kwargs = {
            "api_key": "dummy_key",  # LMStudio doesn't require an API key, but the ChatOpenAI wrapper expects one. We can use a dummy value.
            "base_url": self.config.base_url,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "model_kwargs": {},
        }
        
        if self.config.top_p:
            kwargs["model_kwargs"]["top_p"] = self.config.top_p

        return ChatOpenAI(**kwargs)
