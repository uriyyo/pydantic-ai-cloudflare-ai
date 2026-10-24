import pytest
from pydantic_ai import Agent, Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import infer_model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.typesafe import TypeSafeModel

import pydantic_ai_cloudflare_ai


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "gw")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk")
    pydantic_ai_cloudflare_ai.install()
    pydantic_ai_cloudflare_ai.install()  # idempotent


def test_agent_model_strings():
    assert isinstance(Agent("cloudflare/workers-ai:@cf/meta/llama-3.3-70b-instruct-fp8-fast").model, OpenAIChatModel)
    assert isinstance(Agent("cloudflare/anthropic:claude-sonnet-4-5").model, AnthropicModel)
    assert isinstance(Agent("cloudflare/workers-ai:typesafe/jev").model, TypeSafeModel)


def test_embedder_model_string():
    assert isinstance(
        Embedder("cloudflare/workers-ai:@cf/baai/bge-m3", defer_model_check=False).model, OpenAIEmbeddingModel
    )


def test_other_ids_fall_through():
    assert isinstance(infer_model("test"), TestModel)
    with pytest.raises(UserError, match="Unknown model"):
        infer_model("nope:x")
