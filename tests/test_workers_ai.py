import base64
import json

import httpx2
import pytest
import respx
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import BinaryAudio
from pydantic_ai.models.openai import OpenAIChatModel

from pydantic_ai_cloudflare_ai import WorkersAI, workers_ai_embedding_model, workers_ai_model

API = "https://api.cloudflare.com/client/v4/accounts/acct/ai"

WAI = WorkersAI(
    account_id="acct",
    api_token="cf-token",
    gateway_id="gw",
)


def test_chat_model():
    model = WAI.model("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    assert model.client.base_url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/"
    assert model.client._client.headers["cf-aig-gateway-id"] == "gw"  # noqa: SLF001


def test_embedding_model():
    model = WAI.embedding_model("@cf/baai/bge-m3")
    assert isinstance(model, OpenAIEmbeddingModel)
    assert model.model_name == "@cf/baai/bge-m3"


def _workers_ai(**kwargs):
    return WorkersAI(account_id="acct", api_token="cf-token", gateway_id=None, **kwargs)


async def test_tts_binary_response(httpx2_mock: respx.Router):
    route = httpx2_mock.post(f"{API}/run/@cf/deepgram/aura-1").respond(
        content=b"MP3",
        headers={"content-type": "audio/mpeg"},
    )

    audio = await _workers_ai().tts.run(
        "@cf/deepgram/aura-1",
        "hi",
        params={"speaker": "luna"},
    )

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer cf-token"
    assert "cf-aig-gateway-id" not in request.headers
    assert json.loads(request.content) == {"text": "hi", "speaker": "luna"}
    assert audio.data == b"MP3"
    assert audio.media_type == "audio/*"  # the run endpoint drops the header and these bytes show no format


async def test_tts_json_response(httpx2_mock: respx.Router):
    route = httpx2_mock.post(f"{API}/run/@cf/myshell-ai/melotts").respond(
        json={"result": {"audio": base64.b64encode(b"ID3mp3").decode()}, "success": True},
    )

    audio = await _workers_ai().tts.run(
        "@cf/myshell-ai/melotts",
        "hi",
        params={"lang": "en"},
    )

    assert json.loads(route.calls.last.request.content) == {"prompt": "hi", "lang": "en"}
    assert audio.data == b"ID3mp3"
    assert audio.media_type == "audio/mpeg"


def test_missing_env(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    with pytest.raises(UserError, match="CLOUDFLARE_API_TOKEN"):
        WorkersAI(account_id="acct")


def test_shortcuts(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    assert isinstance(workers_ai_model("@cf/meta/llama-3.3-70b-instruct-fp8-fast"), OpenAIChatModel)
    assert isinstance(workers_ai_embedding_model("@cf/baai/bge-m3"), OpenAIEmbeddingModel)


async def test_run_third_party_model_uses_body_form(httpx2_mock: respx.Router):
    route = httpx2_mock.post(f"{API}/run").respond(json={"result": {"ok": True}, "success": True})

    assert await _workers_ai().run("typesafe/jev", {"state": "x"}) == {"ok": True}
    assert json.loads(route.calls.last.request.content) == {"model": "typesafe/jev", "input": {"state": "x"}}


async def test_run_timeout_is_configurable(httpx2_mock: respx.Router):
    route = httpx2_mock.post(f"{API}/run/@cf/x").respond(json={"result": {}, "success": True})
    wai = _workers_ai(http_client=httpx2.AsyncClient(timeout=5))

    await wai.run("@cf/x", {}, timeout=7)
    await wai.run("@cf/x", {})

    assert route.calls[0].request.extensions["timeout"]["read"] == 7
    assert route.calls[1].request.extensions["timeout"]["read"] == 5  # the client's own default applies


async def test_tts_stream_yields_chunks_as_they_arrive(httpx2_mock: respx.Router):
    async def chunks():
        yield b"MP"
        yield b"3"

    route = httpx2_mock.post(f"{API}/run/@cf/deepgram/aura-1").respond(
        stream=_ByteStream(chunks()),
        headers={"content-type": "audio/mpeg"},
    )

    async with _workers_ai().tts.stream(
        "@cf/deepgram/aura-1",
        "hi",
        params={"speaker": "luna"},
    ) as audio:
        out = [c async for c in audio]

    assert json.loads(route.calls.last.request.content) == {"text": "hi", "speaker": "luna"}
    assert audio.media_type == "audio/mpeg"
    assert out == [b"MP", b"3"]


class _ByteStream(httpx2.AsyncByteStream):
    def __init__(self, source):
        self._source = source

    async def __aiter__(self):
        async for chunk in self._source:
            yield chunk


def test_tts_is_cached_per_instance():
    assert WAI.tts is WAI.tts
    assert WAI.tts.client is WAI


async def test_tts_third_party_downloads_audio_url_without_auth(httpx2_mock: respx.Router):
    storage = httpx2_mock.get("https://cdn.example/a.mp3").respond(
        content=b"MP3",
        headers={"content-type": "audio/mpeg"},
    )
    run = httpx2_mock.post(f"{API}/run").respond(
        json={"result": {"state": "Completed", "result": {"audio": "https://cdn.example/a.mp3"}}, "success": True},
    )

    wai = _workers_ai()
    params = {"voice": "x", "unlisted": 1}
    assert await wai.tts.run("elevenlabs/some-new-model", "hi", params=params) == BinaryAudio(
        b"MP3", media_type="audio/mpeg"
    )
    async with wai.tts.stream("elevenlabs/some-new-model", "hi", params=params) as audio:
        assert audio.media_type == "audio/mpeg"
        assert [c async for c in audio] == [b"MP3"]

    assert json.loads(run.calls.last.request.content) == {
        "model": "elevenlabs/some-new-model",
        "input": {"text": "hi", "voice": "x", "unlisted": 1},
    }
    assert "Authorization" not in storage.calls.last.request.headers


async def test_tts_stream_reads_json_answer_whole(httpx2_mock: respx.Router):
    route = httpx2_mock.post(f"{API}/run/@cf/myshell-ai/melotts").respond(
        json={"result": {"audio": base64.b64encode(b"MP3").decode()}, "success": True},
    )

    async with _workers_ai().tts.stream("@cf/myshell-ai/melotts", "hi", params={"lang": "en"}) as audio:
        assert [c async for c in audio] == [b"MP3"]

    assert json.loads(route.calls.last.request.content) == {"prompt": "hi", "lang": "en"}


async def test_tts_data_uri_answer(httpx2_mock: respx.Router):
    wav = base64.b64encode(b"RIFF\0\0\0\0WAVE").decode()
    httpx2_mock.post(f"{API}/run").respond(json={"result": {"audio": f"data:audio/wav;base64,{wav}"}, "success": True})

    audio = await _workers_ai().tts.run("google/gemini-3.8-flash-tts", "hi", params={"voice": "Kore"})

    assert audio == BinaryAudio(b"RIFF\0\0\0\0WAVE", media_type="audio/wav")


def test_model_routes_by_id():
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIResponsesModel

    from pydantic_ai_cloudflare_ai.workers_ai import WorkersAIChatModel

    assert isinstance(WAI.model("@cf/meta/llama-3.3-70b-instruct-fp8-fast"), WorkersAIChatModel)

    chat = WAI.model("google/gemini-2.5-flash")
    assert type(chat) is OpenAIChatModel
    assert chat.base_url == f"{API}/v1/"

    responses = WAI.model("openai/gpt-5-mini", api="responses")
    assert isinstance(responses, OpenAIResponsesModel)
    assert responses.base_url == f"{API}/v1/"

    anthropic = WAI.model("anthropic/claude-haiku-4.5")
    assert isinstance(anthropic, AnthropicModel)
    assert anthropic.base_url == f"{API}/"


async def test_anthropic_messages_go_without_query_string(httpx2_mock: respx.Router):
    from pydantic_ai import Agent

    route = httpx2_mock.post(f"{API}/v1/messages").respond(
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "pong"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )

    result = await Agent(_workers_ai().model("anthropic/claude-haiku-4.5")).run("ping")

    request = route.calls.last.request
    assert result.output == "pong"
    assert request.url.query == b""
    assert request.headers["Authorization"] == "Bearer cf-token"
    assert json.loads(request.content)["model"] == "anthropic/claude-haiku-4.5"
