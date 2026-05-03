import sys
from types import SimpleNamespace
from pathlib import Path
from typing import List

import dspy
import pytest

src_path = Path(__file__).resolve().parent.parent

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.llm import LLM
import src.llm.lmstudio as lmstudio_module


class FlexibleMarkerSignature(dspy.Signature):
    candidate_classes: List[str] = dspy.OutputField()
    entitys: List[str] = dspy.OutputField()


class FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeDspyLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def patch_lmstudio_clients(monkeypatch):
    configure_calls = []
    monkeypatch.setattr(lmstudio_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(lmstudio_module.dspy, "LM", FakeDspyLM)
    monkeypatch.setattr(
        lmstudio_module.dspy,
        "settings",
        SimpleNamespace(configure=lambda **kwargs: configure_calls.append(kwargs)),
    )
    return configure_calls


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
    assert type(patch_lmstudio_clients[-1]["adapter"]).__name__ == "FlexibleChatAdapter"
    assert patch_lmstudio_clients[-1]["adapter"].use_json_adapter_fallback is False


def test_lmstudio_dspy_default_adapter_skips_explicit_adapter(patch_lmstudio_clients):
    LLM(
        "lmstudio",
        "dspy",
        model="llama-3.3-70b-instruct",
        base_url="http://localhost:1234/v1",
        dspy_adapter="default",
    )

    assert "adapter" not in patch_lmstudio_clients[-1]


def test_lmstudio_flexible_chat_adapter_parses_compact_field_markers():
    adapter = lmstudio_module.FlexibleChatAdapter(use_json_adapter_fallback=False)

    parsed = adapter.parse(
        FlexibleMarkerSignature,
        '[[##candidate_classes##]]\n["provone:Program"]\n\n'
        '[[##entitys##]]\n["ChatBS-NexGen:query_result_post_processor"]\n\n'
        '[[##completed##]]',
    )

    assert parsed == {
        "candidate_classes": ["provone:Program"],
        "entitys": ["ChatBS-NexGen:query_result_post_processor"],
    }


def test_lmstudio_dspy_strips_legacy_lm_studio_prefix(patch_lmstudio_clients):
    llm = LLM(
        "lmstudio",
        "dspy",
        model="lm_studio/qwen/qwen3.5-4b",
        base_url="http://localhost:1234/v1",
    )

    assert llm.llm.kwargs["model"] == "openai/qwen/qwen3.5-4b"


def test_lmstudio_dspy_strips_legacy_lmstudio_prefix(patch_lmstudio_clients):
    llm = LLM(
        "lmstudio",
        "dspy",
        model="lmstudio/llama-3.3-70b-instruct",
        base_url="http://localhost:1234/v1",
    )

    assert llm.llm.kwargs["model"] == "openai/llama-3.3-70b-instruct"
