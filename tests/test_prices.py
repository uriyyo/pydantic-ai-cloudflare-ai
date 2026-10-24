"""Cost lookup works without custom genai-prices registration: genai-prices knows Workers AI by URL and
TypeSafe by id, and gateway routes fall back to their native provider ids."""

import pytest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.usage import RequestUsage

from pydantic_ai_cloudflare_ai import AIGateway, WorkersAI

WAI = WorkersAI(account_id="acct", api_token="cf-token")
GW = AIGateway(
    account_id="acct",
    gateway_id="gw",
    token="gw-token",
)


def _cost(model: Model, model_name: str | None = None):
    response = ModelResponse(
        parts=[],
        usage=RequestUsage(input_tokens=1_000_000, output_tokens=1_000_000),
        model_name=model_name or model.model_name,
        provider_name=model.system,
        provider_url=model.base_url,
    )
    return response.cost()


@pytest.mark.parametrize(
    ("model", "model_name", "provider"),
    [
        (WAI.model("@cf/meta/llama-3.3-70b-instruct-fp8-fast"), None, "cloudflare"),
        (WAI.decision_model(), "jev-1.13.0", "typesafe"),
        (GW.model("openai/gpt-5.6"), None, "openai"),
        (GW.model("anthropic/claude-sonnet-4-5"), None, "anthropic"),
        (GW.model("google-ai-studio/gemini-2.5-flash"), None, "google"),
    ],
)
def test_cost_resolves(
    model,
    model_name,
    provider,
):
    cost = _cost(model, model_name)
    assert cost.provider.id == provider
    assert cost.total_price > 0
