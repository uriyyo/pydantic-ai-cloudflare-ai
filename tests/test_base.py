import httpx2
import respx

from pydantic_ai_cloudflare_ai import AIGateway, CloudflareAI, GatewayOptions, WorkersAI


def test_both_implement_the_interface():
    cfs: list[CloudflareAI] = [
        WorkersAI(
            account_id="acct",
            api_token="cf-token",
            gateway_id="gw",
        ),
        AIGateway(
            account_id="acct",
            gateway_id="gw",
            token="gw-token",
        ),
    ]
    for cf in cfs:
        assert cf.url("x").startswith("https://")
        sent = dict(cf.http().headers)
        assert {k.lower(): v for k, v in cf.headers().items()}.items() <= sent.items()


def test_env_var_names_keep_their_case(monkeypatch):
    from pydantic_ai_cloudflare_ai._utils import CloudflareEnvVar

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    assert CloudflareEnvVar.CLOUDFLARE_ACCOUNT_ID == "CLOUDFLARE_ACCOUNT_ID"
    assert CloudflareEnvVar.CLOUDFLARE_ACCOUNT_ID.get() == "acct"


async def test_shared_http_client_keeps_instances_apart(httpx2_mock: respx.Router):
    route = httpx2_mock.post("https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/x").respond(
        json={"result": {}},
    )
    client = httpx2.AsyncClient(headers={"x-app": "mine"})
    a = WorkersAI(
        account_id="acct",
        api_token="A",
        gateway_id="gw",
        http_client=client,
        options=GatewayOptions(skip_cache=True),
    )
    b = WorkersAI(account_id="acct", api_token="B", gateway_id=None, http_client=client)

    await a.run("@cf/x", {})
    await b.run("@cf/x", {})
    await a.run("@cf/x", {})

    seen = [call.request.headers for call in route.calls]
    assert [h["authorization"] for h in seen] == ["Bearer A", "Bearer B", "Bearer A"]
    assert "cf-aig-skip-cache" not in seen[1]
    assert "cf-aig-gateway-id" not in seen[1]
    assert all(h["x-app"] == "mine" for h in seen)
    assert "authorization" not in client.headers
    assert not client.is_closed


async def test_own_http_client_is_closed():
    wai = WorkersAI(account_id="acct", api_token="cf-token")

    async with wai.http() as client:
        pass
    assert client.is_closed
