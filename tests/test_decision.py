import json

import httpx2
import pytest
import respx
from pydantic_ai.models.typesafe import TypeSafeModel
from typesafe_sdk import AsyncTypeSafeClient

from pydantic_ai_cloudflare_ai import WorkersAI
from pydantic_ai_cloudflare_ai.workers_ai.decision import RunTransport

RUN_URL = "https://api.cloudflare.com/client/v4/accounts/acct/ai/run"
ANSWER = {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "sales": 0.1}, "confidence": 0.8}


def test_decision_model():
    model = WorkersAI(account_id="acct", api_token="cf-token").decision_model()
    assert isinstance(model, TypeSafeModel)
    assert model.model_name == "typesafe/jev"


async def test_transport_rewrites_request_and_unwraps_response(httpx2_mock: respx.Router):
    result = {"model": "jev-1.13.0", "answers": {"dept": ANSWER}, "usage": {"input_tokens": 3, "output_tokens": 1}}
    route = httpx2_mock.post(RUN_URL).respond(
        json={"result": {"state": "charged twice", "result": result}, "success": True},
    )

    http = httpx2.AsyncClient(transport=RunTransport(RUN_URL, "typesafe/jev"), headers={"cf-aig-gateway-id": "gw"})
    client = AsyncTypeSafeClient(
        api_key="cf-token",
        base_url=RUN_URL,
        http_client=http,
    )
    response = await client.system_one(
        "charged twice",
        {"dept": {"type": "choice", "instructions": "Which department?", "criteria": {"billing": None, "sales": None}}},
    )

    assert response.choices["dept"].choice == "billing"
    assert response.model == "jev-1.13.0"

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer cf-token"
    assert request.headers["cf-aig-gateway-id"] == "gw"
    body = json.loads(request.content)
    assert body == {
        "model": "typesafe/jev",
        "input": {"state": "charged twice", "questions": body["input"]["questions"]},
    }
    assert body["input"]["questions"]["dept"]["type"] == "choice"


async def test_transport_passes_errors_through(httpx2_mock: respx.Router):
    httpx2_mock.post(RUN_URL).respond(401, json={"errors": []})

    client = AsyncTypeSafeClient(
        api_key="bad",
        base_url=RUN_URL,
        http_client=httpx2.AsyncClient(transport=RunTransport(RUN_URL, "typesafe/jev")),
    )
    with pytest.raises(Exception, match="401"):
        await client.system_one("x", {"q": {"type": "noul", "instructions": "ok?"}})
