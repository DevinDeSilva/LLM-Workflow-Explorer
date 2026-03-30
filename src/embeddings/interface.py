from icecream import ic
from typing import Optional

from src.embeddings.base import BaseEmbeddings


class Embeddings:
    def __new__(cls, embedding_type: str, library: str = "langchain", **kwargs):
        return cls.__create_concrete__(embedding_type, library, **kwargs)

    @classmethod
    def __create_concrete__(
        cls,
        embedding_type: str,
        library: str = "langchain",
        **kwargs,
    ) -> Optional[BaseEmbeddings]:
        if embedding_type == "openai":
            from src.config.embeddings.openai import OpenAIEmbeddingsConfig
            from src.embeddings.openai import OpenAIEmbeddingsModel

            config = OpenAIEmbeddingsConfig(**kwargs)
            ic(config)
            return OpenAIEmbeddingsModel(config, library)
        elif embedding_type == "lmstudio":
            from src.config.embeddings.lmstudio import LMStudioEmbeddingsConfig
            from src.embeddings.lmstudio import LMStudioEmbeddingsModel

            config = LMStudioEmbeddingsConfig(**kwargs)
            ic(config)
            return LMStudioEmbeddingsModel(config, library)
        else:
            raise NotImplementedError
