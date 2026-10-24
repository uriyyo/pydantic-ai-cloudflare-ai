import base64
import json

import pytest
import respx

from pydantic_ai_cloudflare_ai import AIGateway
from pydantic_ai_cloudflare_ai.tts import TTSError

GATEWAY = "https://gateway.ai.cloudflare.com/v1/acct/gw"
GEMINI = f"{GATEWAY}/google-ai-studio/v1beta/models/gemini-2.5-flash-preview-tts"

GW = AIGateway(
    account_id="acct",
    gateway_id="gw",
    token="gw-token",
)


def _sse(*payloads):
    return "".join(f"data: {json.dumps(payload)}\r\n\r\n" for payload in payloads)


def _speech(data):
    return {"candidates": [{"content": {"parts": [{"inlineData": {"data": base64.b64encode(data).decode()}}]}}]}


async def test_openai_run(httpx2_mock: respx.Router):
    route = httpx2_mock.post(f"{GATEWAY}/openai/audio/speech").respond(
        content=b"MP3",
        headers={"content-type": "audio/mpeg"},
    )

    audio = await GW.tts.run(
        "openai/gpt-4o-mini-tts",
        "hi",
        params={"voice": "alloy"},
        api_key="sk",
    )

    request = route.calls.last.request
    assert request.headers["cf-aig-authorization"] == "Bearer gw-token"
    assert request.headers["Authorization"] == "Bearer sk"
    assert json.loads(request.content) == {"model": "gpt-4o-mini-tts", "input": "hi", "voice": "alloy"}
    assert audio.data == b"MP3"
    assert audio.media_type == "audio/mpeg"


async def test_openai_stream(httpx2_mock: respx.Router):
    route = httpx2_mock.post(f"{GATEWAY}/openai/audio/speech").respond(
        content=b"MP3",
        headers={"content-type": "audio/mpeg"},
    )

    async with GW.tts.stream("openai/gpt-4o-mini-tts", "hi", params={"voice": "alloy"}) as audio:
        assert audio.media_type == "audio/mpeg"
        assert (await audio.read()).data == b"MP3"

    assert route.calls.last.request.headers["Authorization"] == "Bearer gw-token"


async def test_gemini_run(httpx2_mock: respx.Router):
    route = httpx2_mock.post(f"{GEMINI}:generateContent").respond(json=_speech(b"PCM"))

    audio = await GW.tts.run(
        "google-ai-studio/gemini-2.5-flash-preview-tts",
        "hi",
        params={"voice": "Kore"},
    )

    request = route.calls.last.request
    assert request.headers["x-goog-api-key"] == "gw-token"
    assert json.loads(request.content) == {
        "contents": [{"parts": [{"text": "hi"}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
        },
    }
    assert audio.data == b"PCM"
    assert audio.media_type == "audio/L16;codec=pcm;rate=24000"


async def test_unsupported():
    with pytest.raises(ValueError, match="Unknown AI Gateway TTS route"):
        await GW.tts.run("anthropic/x", "hi")
    with pytest.raises(ValueError, match="Unknown AI Gateway TTS route"):
        async with GW.tts.stream("anthropic/x", "hi"):
            pass


async def test_gemini_stream(httpx2_mock: respx.Router):
    httpx2_mock.post(f"{GEMINI}:streamGenerateContent", params={"alt": "sse"}).respond(
        text=_sse(_speech(b"PC"), _speech(b"M"), {"candidates": [{"finishReason": "STOP"}]}),
        headers={"content-type": "text/event-stream"},
    )

    async with GW.tts.stream("google-ai-studio/gemini-2.5-flash-preview-tts", "hi") as audio:
        assert audio.media_type == "audio/L16;codec=pcm;rate=24000"
        assert [c async for c in audio] == [b"PC", b"M"]


async def test_gemini_without_audio_raises(httpx2_mock: respx.Router):
    blocked = {"candidates": [{"finishReason": "PROHIBITED_CONTENT"}]}
    httpx2_mock.post(f"{GEMINI}:generateContent").respond(json=blocked)
    httpx2_mock.post(f"{GEMINI}:streamGenerateContent").respond(
        text=_sse(blocked),
        headers={"content-type": "text/event-stream"},
    )

    with pytest.raises(TTSError, match="PROHIBITED_CONTENT"):
        await GW.tts.run("google-ai-studio/gemini-2.5-flash-preview-tts", "hi")
    with pytest.raises(TTSError, match="PROHIBITED_CONTENT"):
        async with GW.tts.stream("google-ai-studio/gemini-2.5-flash-preview-tts", "hi") as audio:
            [c async for c in audio]
