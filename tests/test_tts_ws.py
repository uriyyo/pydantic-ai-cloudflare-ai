import base64
import json
from contextlib import asynccontextmanager

import httpx2
import pytest

from pydantic_ai_cloudflare_ai import WorkersAI
from pydantic_ai_cloudflare_ai.tts.workers_ai.frames import GrokFrames, TTSError, frames_for


class FakeSocket:
    def __init__(self, replies):
        self.sent = []
        self.replies = iter(replies)

    async def send_text(self, frame):
        self.sent.append(json.loads(frame))

    async def receive_text(self, timeout=None):  # noqa: ARG002 - mirrors the httpx2 session
        return json.dumps(next(self.replies))


async def test_ws_stream(monkeypatch):
    socket = FakeSocket(
        [
            {"type": "audio.delta", "delta": base64.b64encode(b"MP").decode()},
            {"type": "something.else"},
            {"type": "audio.delta", "delta": base64.b64encode(b"3").decode()},
            {"type": "audio.done"},
        ]
    )
    seen = {}

    @asynccontextmanager
    async def websocket(
        self,
        url,
        *,
        params,
        timeout,
    ):
        seen.update(
            url=url,
            params=params,
            headers=dict(self.headers),
            timeout=timeout,
        )
        yield socket

    monkeypatch.setattr(
        httpx2.AsyncClient,
        "websocket",
        websocket,
    )
    wai = WorkersAI(
        account_id="acct",
        api_token="cf-token",
        gateway_id="gw",
    )
    async with wai.tts.stream(
        "xai/grok-tts", "hi", transport="ws", params={"language": "en", "voice_id": "eve"}
    ) as audio:
        out = [c async for c in audio]

    assert audio.media_type == "audio/mpeg"
    assert out == [b"MP", b"3"]
    assert socket.sent == [{"type": "text.delta", "delta": "hi"}, {"type": "text.done"}]
    assert seen["url"] == "wss://api.cloudflare.com/client/v4/accounts/acct/ai/run"
    assert seen["params"] == {"model": "xai/grok-tts", "websocket": "true", "language": "en", "voice_id": "eve"}
    assert seen["headers"]["authorization"] == "Bearer cf-token"
    assert seen["headers"]["cf-aig-gateway-id"] == "gw"
    assert seen["timeout"] is httpx2.USE_CLIENT_DEFAULT


def test_grok_error_frame():
    with pytest.raises(TTSError, match="boom"):
        GrokFrames().receive(json.dumps({"type": "error", "message": "boom"}))


def test_unknown_model_needs_explicit_frames():
    with pytest.raises(ValueError, match="frames="):
        frames_for("@cf/deepgram/aura-1")
