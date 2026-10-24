from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai_cloudflare_ai import urls
from pydantic_ai_cloudflare_ai._utils import read_timeout, resolve_timeout
from pydantic_ai_cloudflare_ai.tts.audio import AudioStream, binary_audio
from pydantic_ai_cloudflare_ai.tts.http import content_type, download, post_stream
from pydantic_ai_cloudflare_ai.workers_ai.schemas import RunResult

from .frames import TTSFrames, frames_for
from .schemas import AudioResponse, TTSParams, TTSRequest, WSQuery

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from httpx2.websockets import AsyncWebSocketSession
    from pydantic import JsonValue
    from pydantic_ai.messages import BinaryAudio

    from pydantic_ai_cloudflare_ai.types import Timeout, TTSTransport
    from pydantic_ai_cloudflare_ai.workers_ai.client import WorkersAI

__all__ = ["TTS"]


@dataclass(frozen=True)
class TTS:
    client: WorkersAI

    def _map_request(
        self,
        model_id: str,
        text: str,
        params: TTSParams | None,
    ) -> TTSRequest:
        _params: dict[str, Any] = dict(params or {})

        if "melotts" in model_id:
            _params["prompt"] = text
        else:
            _params["text"] = text

        return TTSRequest.model_validate(_params)

    async def run(
        self,
        model_id: str,
        text: str,
        *,
        params: TTSParams | None = None,
        timeout: Timeout = None,
    ) -> BinaryAudio:
        request = self._map_request(
            model_id,
            text,
            params,
        )

        match await self.client.run(
            model_id,
            request,
            timeout=timeout,
        ):
            case bytes() as audio:
                return binary_audio(audio)
            case result:
                return await self._audio(result, timeout)

    async def _audio(self, result: JsonValue, timeout: Timeout) -> BinaryAudio:
        match result:
            case {"result": {"audio": _} as inner}:
                # third-party models wrap their result once more: `{"state": "Completed", "result": {...}}`.
                audio = AudioResponse.model_validate(inner).audio
            case _:
                audio = AudioResponse.model_validate(result).audio

        if audio.startswith(("https://", "http://")):
            response = await download(audio, timeout=timeout)
            return binary_audio(response.content, content_type(response))

        if audio.startswith("data:"):
            header, _, data = audio.partition(",")
            return binary_audio(
                base64.b64decode(data),
                header.removeprefix("data:").split(";")[0],
            )

        return binary_audio(base64.b64decode(audio))

    @asynccontextmanager
    async def stream(
        self,
        model_id: str,
        text: str,
        *,
        transport: TTSTransport = "http",
        frames: TTSFrames | None = None,
        params: TTSParams | None = None,
        timeout: Timeout = None,
    ) -> AsyncGenerator[AudioStream]:
        options = dict(params or {})

        match transport:
            case "http" if not model_id.startswith("@cf/"):
                # Third-party models answer with whole audio, so it comes as one chunk.
                yield AudioStream.of(await self.run(model_id, text, params=params, timeout=timeout))
            case "http":
                request = self._map_request(model_id, text, params)

                async with post_stream(
                    self.client,
                    self.client.url(f"run/{model_id}"),
                    request.model_dump_json(exclude_unset=True),
                    timeout=timeout,
                ) as response:
                    # Only some `@cf` models (Aura) stream audio; the rest answer JSON, read whole.
                    if (content_type(response) or "").startswith("application/json"):
                        await response.aread()
                        result = RunResult.model_validate_json(response.content).result
                        yield AudioStream.of(await self._audio(result, timeout))
                    else:
                        yield await AudioStream.start(response.aiter_bytes(), content_type(response))
            case "ws":
                query = WSQuery.model_validate({"model": model_id} | options)
                frames = frames or frames_for(model_id)

                async with (
                    self.client.http() as http,
                    http.websocket(
                        urls.websocket(self.client.url("run")),
                        params=query.model_dump(),
                        timeout=resolve_timeout(timeout),
                    ) as socket,
                ):
                    for frame in frames.send(text):
                        await socket.send_text(frame)

                    yield await AudioStream.start(self._receive(socket, frames, timeout), frames.media_type)

    async def _receive(
        self,
        socket: AsyncWebSocketSession,
        frames: TTSFrames,
        timeout: Timeout,
    ) -> AsyncIterator[bytes]:
        while True:
            message = await socket.receive_text(timeout=read_timeout(timeout))
            audio = frames.receive(message)

            if audio is None:
                return
            if audio:
                yield audio
