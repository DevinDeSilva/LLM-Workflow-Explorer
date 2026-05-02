from src.llm.base import BaseLLM
from src.config.llm.lmstudio import LMStudioConfig
from icecream import ic
import dspy
import re
from dspy.utils.exceptions import AdapterParseError


try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install langchain_openai")


class FlexibleChatAdapter(dspy.ChatAdapter):
    def parse(self, signature, completion):
        try:
            return super().parse(signature, completion)
        except AdapterParseError:
            normalized_completion = re.sub(
                r"\[\[\s*##\s*(\w+)\s*##\s*\]\]",
                r"[[ ## \1 ## ]]",
                completion,
            )
            if normalized_completion == completion:
                raise
            return super().parse(signature, normalized_completion)


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

    def _dspy_adapter(self):
        adapter_name = (self.config.dspy_adapter or "chat").strip().lower()
        if adapter_name == "default":
            return None
        if adapter_name == "json":
            return dspy.JSONAdapter()
        if adapter_name == "chat":
            return FlexibleChatAdapter(
                use_json_adapter_fallback=self.config.dspy_json_adapter_fallback
            )
        raise ValueError(f"Unsupported DSPy adapter for LMStudio: {adapter_name}")

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

            adapter = self._dspy_adapter()
            configure_kwargs = {"lm": lm, "trace": []}
            if adapter is not None:
                configure_kwargs["adapter"] = adapter

            # Register it globally
            dspy.settings.configure(**configure_kwargs)
            return lm
        
        else:
            raise ValueError("Not the correct {}".format(self.library))
