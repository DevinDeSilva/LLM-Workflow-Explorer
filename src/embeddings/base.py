from typing import List, Sequence


class BaseEmbeddings:
    def __init__(self, config, library: str = "langchain"):
        self.config = config
        self.library = library
        self.embeddings = self._create_client()

    def _create_client(self):
        raise NotImplementedError

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if self.library != "langchain":
            raise ValueError(f"Embeddings are not implemented for library `{self.library}`.")

        normalized_texts = list(texts)
        if not normalized_texts:
            return []

        return self.embeddings.embed_documents(normalized_texts)

    def embed_query(self, text: str) -> List[float]:
        if self.library != "langchain":
            raise ValueError(f"Embeddings are not implemented for library `{self.library}`.")

        return self.embeddings.embed_query(text)

    async def aembed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if self.library != "langchain":
            raise ValueError(f"Embeddings are not implemented for library `{self.library}`.")

        normalized_texts = list(texts)
        if not normalized_texts:
            return []

        return await self.embeddings.aembed_documents(normalized_texts)

    async def aembed_query(self, text: str) -> List[float]:
        if self.library != "langchain":
            raise ValueError(f"Embeddings are not implemented for library `{self.library}`.")

        return await self.embeddings.aembed_query(text)
