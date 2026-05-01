import sys
from pathlib import Path

import pytest

src_path = Path(__file__).resolve().parent.parent

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.llm import LLM
import src.llm.lmstudio as lmstudio_module


class FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeDspyLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def patch_lmstudio_clients(monkeypatch):
    monkeypatch.setattr(lmstudio_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(lmstudio_module.dspy, "LM", FakeDspyLM)
    monkeypatch.setattr(
        lmstudio_module.dspy.settings,
        "configure",
        lambda **kwargs: None,
    )


def test_lmstudio_langchain_uses_raw_model_name(patch_lmstudio_clients):
    llm = LLM(
        "lmstudio",
        "langchain",
        model="llama-3.3-70b-instruct",
        base_url="http://localhost:1234/v1",
        temperature=0,
        max_tokens=300,
    )

    assert llm.llm.kwargs["model"] == "llama-3.3-70b-instruct"
    assert llm.llm.kwargs["base_url"] == "http://localhost:1234/v1"


def test_lmstudio_dspy_prefixes_openai_provider_for_litellm(patch_lmstudio_clients):
    llm = LLM(
        "lmstudio",
        "dspy",
        model="llama-3.3-70b-instruct",
        base_url="http://localhost:1234/v1",
        temperature=0,
        max_tokens=300,
    )

    assert llm.llm.kwargs["model"] == "openai/llama-3.3-70b-instruct"
    assert llm.llm.kwargs["api_base"] == "http://localhost:1234/v1"
    assert "base_url" not in llm.llm.kwargs


def test_lmstudio_dspy_strips_legacy_lm_studio_prefix(patch_lmstudio_clients):
    llm = LLM(
        "lmstudio",
        "dspy",
        model="lm_studio/qwen/qwen3.5-4b",
        base_url="http://localhost:1234/v1",
    )

    assert llm.llm.kwargs["model"] == "openai/qwen/qwen3.5-4b"
