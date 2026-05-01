from src.llm.base import BaseLLM
from src.config.llm.lmstudio import LMStudioConfig
from icecream import ic
import dspy


try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install langchain_openai")

class LMStudio(BaseLLM):
    def __init__(self, config:LMStudioConfig, library:str):
        super().__init__(config, library)
        
    def _model_name(self) -> str:
        return self.config.model.removeprefix("lm_studio/")

    def _dspy_model_name(self) -> str:
        model_name = self._model_name()
        if model_name.startswith("openai/"):
            return model_name
        return f"openai/{model_name}"

    def _create_client(self):
        kwargs = {
            "api_key": "dummy_key",  # LMStudio doesn't require an API key, but the ChatOpenAI wrapper expects one. We can use a dummy value.
            "base_url": self.config.base_url,
            "model": self._model_name(),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        if self.config.top_p:
            kwargs["top_p"] = self.config.top_p
        
        if self.library == "langchain":
            return ChatOpenAI(**kwargs)
        elif self.library == "dspy":
            kwargs["api_base"] = self.config.base_url
            kwargs["model"] = self._dspy_model_name()
            del kwargs["base_url"]
            
            # Create OpenAI LLM wrapper
            lm: dspy.LM = dspy.LM(
                **kwargs
            )

            # Register it globally
            dspy.settings.configure(lm=lm, trace=[])
            return lm
        
        else:
            raise ValueError("Not the correct {}".format(self.library))
