from src.config.embeddings.lmstudio import LMStudioEmbeddingsConfig
from src.embeddings.base import BaseEmbeddings

try:
    from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install langchain_openai")


class LMStudioEmbeddingsModel(BaseEmbeddings):
    def __init__(self, config: LMStudioEmbeddingsConfig, library: str = "langchain"):
        super().__init__(config, library)

    def _create_client(self):
        kwargs = {
            "api_key": "dummy_key",
            "base_url": self.config.base_url,
            "model": self.config.model,
        }
        if self.config.dimensions is not None:
            kwargs["dimensions"] = self.config.dimensions

        if self.library == "langchain":
            return LangChainOpenAIEmbeddings(**kwargs)

        raise ValueError(f"Embeddings are not implemented for library `{self.library}`.")
