from icecream import ic 
from typing import Optional
from src.llm.base import BaseLlm

class LLM:
    def __new__(cls, llm_type, **kwargs):
        """Initialize a base LLM class

        :param config: LLM configuration option class, defaults to None
        :type config: Optional[BaseLlmConfig], optional
        """
        
        return cls.__create_concrete__(llm_type, **kwargs)
        
    @classmethod
    def __create_concrete__(cls, llm_type, **kwargs)  -> Optional[BaseLlm]:
        if llm_type == "openai":
            from src.llm.openai import OpenAILlm
            from src.config.llm.openai import OpenAILlmConfig
            config = OpenAILlmConfig(**kwargs)
            ic(config)
            return OpenAILlm(config)
        elif llm_type == "llmstudio":
            from src.llm.lmstudio import LMStudio
            from src.config.llm.lmstudio import LMStudioConfig
            config = LMStudioConfig(**kwargs)
            ic(config)
            return LMStudio(config)
        else:
            raise NotImplementedError
    
    