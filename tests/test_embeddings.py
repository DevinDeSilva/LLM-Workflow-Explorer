import asyncio
import sys
from pathlib import Path

import pytest

src_path = Path(__file__).resolve().parent.parent

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.config.embeddings.lmstudio import LMStudioEmbeddingsConfig
from src.config.embeddings.openai import OpenAIEmbeddingsConfig
from src.embeddings import Embeddings
import src.embeddings.lmstudio as lmstudio_module
import src.embeddings.openai as openai_module


class FakeLangChainEmbeddings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.embed_documents_calls = []
        self.embed_query_calls = []
        self.aembed_documents_calls = []
        self.aembed_query_calls = []

    def embed_documents(self, texts):
        self.embed_documents_calls.append(list(texts))
        return [[float(index), float(index + 1)] for index, _ in enumerate(texts)]

    def embed_query(self, text):
        self.embed_query_calls.append(text)
        return [0.1, 0.2, 0.3]

    async def aembed_documents(self, texts):
        self.aembed_documents_calls.append(list(texts))
        return [[float(index), float(index + 1)] for index, _ in enumerate(texts)]

    async def aembed_query(self, text):
        self.aembed_query_calls.append(text)
        return [0.4, 0.5, 0.6]


@pytest.fixture
def patch_embedding_clients(monkeypatch):
    monkeypatch.setattr(openai_module, "LangChainOpenAIEmbeddings", FakeLangChainEmbeddings)
    monkeypatch.setattr(lmstudio_module, "LangChainOpenAIEmbeddings", FakeLangChainEmbeddings)


def test_openai_embeddings_create_client_uses_config_values(patch_embedding_clients):
    model = Embeddings(
        "openai",
        api_key="test-openai-key",
        model="text-embedding-3-small",
        dimensions=512,
    )

    assert model.config == OpenAIEmbeddingsConfig(
        api_key="test-openai-key",
        model="text-embedding-3-small",
        dimensions=512,
    )
    assert isinstance(model.embeddings, FakeLangChainEmbeddings)
    assert model.embeddings.kwargs == {
        "openai_api_key": "test-openai-key",
        "model": "text-embedding-3-small",
        "dimensions": 512,
    }


def test_openai_embeddings_omits_dimensions_when_not_provided(patch_embedding_clients):
    model = Embeddings(
        "openai",
        api_key="test-openai-key",
        model="text-embedding-3-large",
    )

    assert model.embeddings.kwargs == {
        "openai_api_key": "test-openai-key",
        "model": "text-embedding-3-large",
    }


def test_lmstudio_embeddings_create_client_uses_config_values(patch_embedding_clients):
    model = Embeddings(
        "lmstudio",
        model="nomic-embed-text",
        base_url="http://localhost:1234/v1",
        dimensions=768,
    )

    assert model.config == LMStudioEmbeddingsConfig(
        model="nomic-embed-text",
        base_url="http://localhost:1234/v1",
        dimensions=768,
    )
    assert isinstance(model.embeddings, FakeLangChainEmbeddings)
    assert model.embeddings.kwargs == {
        "api_key": "dummy_key",
        "base_url": "http://localhost:1234/v1",
        "model": "nomic-embed-text",
        "check_embedding_ctx_length": False,
        "dimensions": 768,
    }


def test_embed_documents_and_query_delegate_to_client(patch_embedding_clients):
    model = Embeddings("openai", api_key="test-openai-key")

    document_embeddings = model.embed_documents(["alpha", "beta"])
    query_embedding = model.embed_query("alpha")

    assert document_embeddings == [[0.0, 1.0], [1.0, 2.0]]
    assert query_embedding == [0.1, 0.2, 0.3]
    assert model.embeddings.embed_documents_calls == [["alpha", "beta"]]
    assert model.embeddings.embed_query_calls == ["alpha"]


def test_async_embed_methods_delegate_to_client(patch_embedding_clients):
    model = Embeddings("lmstudio")

    document_embeddings = asyncio.run(model.aembed_documents(["alpha", "beta"]))
    query_embedding = asyncio.run(model.aembed_query("beta"))

    assert document_embeddings == [[0.0, 1.0], [1.0, 2.0]]
    assert query_embedding == [0.4, 0.5, 0.6]
    assert model.embeddings.aembed_documents_calls == [["alpha", "beta"]]
    assert model.embeddings.aembed_query_calls == ["beta"]


def test_embed_documents_returns_empty_list_for_empty_input(patch_embedding_clients):
    model = Embeddings("openai", api_key="test-openai-key")

    assert model.embed_documents([]) == []
    assert asyncio.run(model.aembed_documents([])) == []
    assert model.embeddings.embed_documents_calls == []
    assert model.embeddings.aembed_documents_calls == []


def test_openai_embeddings_reject_unsupported_library(patch_embedding_clients):
    with pytest.raises(ValueError, match="Embeddings are not implemented for library `dspy`"):
        Embeddings("openai", library="dspy", api_key="test-openai-key")


def test_lmstudio_embeddings_reject_unsupported_library(patch_embedding_clients):
    with pytest.raises(ValueError, match="Embeddings are not implemented for library `dspy`"):
        Embeddings("lmstudio", library="dspy")


def test_embeddings_factory_rejects_unknown_type():
    with pytest.raises(NotImplementedError):
        Embeddings("unknown-provider")
