from src.config.embeddings.openai import OpenAIEmbeddingsConfig
from src.embeddings.base import BaseEmbeddings

try:
    from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install langchain_openai")


class OpenAIEmbeddingsModel(BaseEmbeddings):
    def __init__(self, config: OpenAIEmbeddingsConfig, library: str = "langchain"):
        super().__init__(config, library)

    def _create_client(self):
        kwargs = {
            "openai_api_key": self.config.api_key,
            "model": self.config.model,
        }
        if self.config.dimensions is not None:
            kwargs["dimensions"] = self.config.dimensions

        if self.library == "langchain":
            return LangChainOpenAIEmbeddings(**kwargs)

        raise ValueError(f"Embeddings are not implemented for library `{self.library}`.")
