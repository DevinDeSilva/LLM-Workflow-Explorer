from icecream import ic 
from src.llm.base import BaseLLM

class LLM:
    def __new__(cls, llm_type, library="dspy", **kwargs):
        """Initialize a base LLM class

        :param config: LLM configuration option class, defaults to None
        :type config: Optional[BaseLLMConfig], optional
        """
        
        return cls.__create_concrete__(llm_type, library, **kwargs)
        
    @classmethod
    def __create_concrete__(cls, llm_type, library,**kwargs)  -> BaseLLM:
        if llm_type == "openai":
            from src.llm.openai import OpenAILlm
            from src.config.llm.openai import OpenAILlmConfig
            config = OpenAILlmConfig(**kwargs)
            ic(config)
            return OpenAILlm(config, library)
        elif llm_type == "lmstudio":
            from src.llm.lmstudio import LMStudio
            from src.config.llm.lmstudio import LMStudioConfig
            config = LMStudioConfig(**kwargs)
            ic(config)
            return LMStudio(config, library)
        else:
            raise NotImplementedError
    
    