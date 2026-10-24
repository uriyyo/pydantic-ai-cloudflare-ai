import pytest
from pydantic import ValidationError

from pydantic_ai_cloudflare_ai import AIGateway, urls


def test_join_and_websocket():
    url = urls.join(
        "https://api.cloudflare.com/client/v4/accounts/",
        "acct",
        "ai",
        "run",
    )
    assert url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run"
    assert urls.websocket(url) == "wss://api.cloudflare.com/client/v4/accounts/acct/ai/run"
    assert urls.websocket("http://localhost:8787/ai/run?x=1") == "ws://localhost:8787/ai/run?x=1"


def test_api_url_is_validated():
    assert urls.validate_http("https://gateway.ai.cloudflare.com") == "https://gateway.ai.cloudflare.com"
    with pytest.raises(ValidationError):
        AIGateway(
            account_id="acct",
            gateway_id="gw",
            api_url="not a url",
        )
