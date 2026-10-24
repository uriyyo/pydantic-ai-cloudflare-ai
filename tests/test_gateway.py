import pytest
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel

from pydantic_ai_cloudflare_ai import AIGateway, gateway_embedding_model, gateway_model

GW = AIGateway(
    account_id="acct",
    gateway_id="gw",
    token="gw-token",
)


def test_openai_route():
    model = GW.model("openai/gpt-5.6", api_key="sk")
    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == "gpt-5.6"
    assert model.client.base_url == "https://gateway.ai.cloudflare.com/v1/acct/gw/openai/"
    assert model.client._client.headers["cf-aig-authorization"] == "Bearer gw-token"  # noqa: SLF001


def test_other_routes():
    anthropic = GW.model("anthropic/claude-sonnet-4-5", api_key="sk")
    assert isinstance(anthropic, AnthropicModel)
    assert str(anthropic.client.base_url) == "https://gateway.ai.cloudflare.com/v1/acct/gw/anthropic/"
    assert isinstance(GW.model("google-ai-studio/gemini-2.5-flash", api_key="sk"), GoogleModel)


def test_embedding_model():
    model = GW.embedding_model("openai/text-embedding-3-small", api_key="sk")
    assert isinstance(model, OpenAIEmbeddingModel)
    assert model.model_name == "text-embedding-3-small"


def test_unknown_route():
    with pytest.raises(ValueError, match="Unknown AI Gateway route"):
        GW.model("xai/grok-4.7")
    with pytest.raises(ValueError, match="Unknown AI Gateway embedding route"):
        GW.embedding_model("anthropic/x")


def test_groq_route():
    from pydantic_ai.models.groq import GroqModel

    model = GW.model("groq/llama-3.3-70b-versatile", api_key="sk")
    assert isinstance(model, GroqModel)
    assert model.client._client.headers["cf-aig-authorization"] == "Bearer gw-token"  # noqa: SLF001


def test_shortcuts(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "gw")
    assert isinstance(gateway_model("openai/gpt-5.6", api_key="sk"), OpenAIResponsesModel)
    assert isinstance(gateway_embedding_model("openai/text-embedding-3-small", api_key="sk"), OpenAIEmbeddingModel)


def test_api_key_defaults_to_gateway_token():
    assert GW.model("openai/gpt-5.6").client.api_key == "gw-token"
    assert GW.model("openai/gpt-5.6", api_key="sk").client.api_key == "sk"


def test_gateway_id_defaults_to_default(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.delenv("CLOUDFLARE_GATEWAY_ID", raising=False)
    assert AIGateway().url("openai") == "https://gateway.ai.cloudflare.com/v1/acct/default/openai"

    monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "gw")
    assert AIGateway().gateway_id == "gw"
