from src.llm.base import BaseLLM
from src.config.llm.openai import OpenAILlmConfig
import dspy

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install langchain_openai")

class OpenAILlm(BaseLLM):
    def __init__(self, config:OpenAILlmConfig, library:str = "langchain"):
        super().__init__(config, library)

    def _create_client(self):
        kwargs = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.top_p: # XXX: need to update
            kwargs["top_p"] = self.config.top_p
        
        if self.library == "langchain":
            return ChatOpenAI(**kwargs)
        elif self.library == "dspy":
            # Create OpenAI LLM wrapper
            lm: dspy.LM = dspy.LM(
                **kwargs
            )

            # Register it globally
            dspy.settings.configure(lm=lm, trace=[])
            return lm
        else:
            raise ValueError("Not the correct {}".format(self.library))


    
