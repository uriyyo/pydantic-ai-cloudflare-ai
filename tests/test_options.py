import json

import pytest
from pydantic import ValidationError

from pydantic_ai_cloudflare_ai import AIGateway, CustomCost, GatewayOptions, WorkersAI


def test_headers():
    options = GatewayOptions(
        cache_ttl=3600,
        cache_key="k",
        skip_cache=True,
        metadata={"team": "AI", "user": 12345},
        collect_log=False,
        request_timeout=5000,
        max_attempts=3,
        retry_delay=1000,
        backoff="exponential",
        custom_cost=CustomCost(per_token_in=0.000001, per_token_out=0.000002),
        byok_alias="prod",
    )

    headers = options.headers()

    assert headers == {
        "cf-aig-cache-ttl": "3600",
        "cf-aig-cache-key": "k",
        "cf-aig-skip-cache": "true",
        "cf-aig-metadata": '{"team":"AI","user":12345}',
        "cf-aig-collect-log": "false",
        "cf-aig-request-timeout": "5000",
        "cf-aig-max-attempts": "3",
        "cf-aig-retry-delay": "1000",
        "cf-aig-backoff": "exponential",
        "cf-aig-custom-cost": headers["cf-aig-custom-cost"],
        "cf-aig-byok-alias": "prod",
    }
    assert json.loads(headers["cf-aig-custom-cost"]) == {"per_token_in": 0.000001, "per_token_out": 0.000002}
    assert GatewayOptions().headers() == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 6},
        {"retry_delay": 60_001},
        {"cache_ttl": 59},
        {"backoff": "random"},
        {"metadata": {str(i): i for i in range(6)}},
        {"metadata": {"cf.user": 1}},
        {"metadata": {"nested": {"a": 1}}},
        {"unknown": 1},
    ],
)
def test_rejects_what_the_gateway_would_drop(kwargs):
    with pytest.raises(ValidationError):
        GatewayOptions(**kwargs)


def test_options_reach_both_clients():
    options = GatewayOptions(skip_cache=True)

    wai = WorkersAI(account_id="acct", api_token="cf-token", gateway_id="gw", options=options)
    gw = AIGateway(account_id="acct", gateway_id="gw", token="gw-token", options=options)

    assert wai.headers()["cf-aig-skip-cache"] == "true"
    assert gw.headers()["cf-aig-skip-cache"] == "true"
    assert gw.model("openai/gpt-5.6").client._client.headers["cf-aig-skip-cache"] == "true"  # noqa: SLF001
