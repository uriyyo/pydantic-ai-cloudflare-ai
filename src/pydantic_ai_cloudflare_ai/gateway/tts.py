from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

from httpx2 import EventSource
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from pydantic_core import to_json

from pydantic_ai_cloudflare_ai._utils import resolve_timeout
from pydantic_ai_cloudflare_ai.headers import CommonHeaders, Header
from pydantic_ai_cloudflare_ai.tts import AudioStream, TTSError, content_type, post_stream
from pydantic_ai_cloudflare_ai.tts.audio import binary_audio

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    import httpx2
    from pydantic_ai.messages import BinaryAudio

    from pydantic_ai_cloudflare_ai.types import Headers, Timeout

    from .client import AIGateway

__all__ = [
    "GatewayTTS",
    "GatewayTTSParams",
    "GeminiTTSParams",
    "OpenAITTSParams",
]


class OpenAITTSParams(TypedDict, total=False):
    voice: str
    response_format: str
    speed: float
    instructions: str


class GeminiTTSParams(TypedDict, total=False):
    voice: str


GatewayTTSParams: TypeAlias = OpenAITTSParams | GeminiTTSParams

GEMINI_MEDIA_TYPE = "audio/L16;codec=pcm;rate=24000"


class _Camel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)


class _InlineData(_Camel):
    data: str


class _Part(_Camel):
    inline_data: _InlineData | None = None


class _Content(_Camel):
    parts: list[_Part] = []


class _Candidate(_Camel):
    content: _Content | None = None
    finish_reason: str | None = None


class GeminiResponse(_Camel):
    # The last streamed chunk may carry only finish reason and usage.
    candidates: list[_Candidate] = []

    def audio(self) -> bytes:
        return b"".join(
            base64.b64decode(part.inline_data.data)
            for candidate in self.candidates
            if candidate.content
            for part in candidate.content.parts
            if part.inline_data
        )

    def finish_reason(self) -> str | None:
        return next((candidate.finish_reason for candidate in self.candidates if candidate.finish_reason), None)


def _no_audio(reason: str | None) -> TTSError:
    # Gemini answers 200 without audio when it blocks or refuses the text.
    return TTSError(f"Gemini returned no audio (finish reason: {reason})")


@dataclass(frozen=True)
class GatewayTTS:
    gateway: AIGateway

    def _request(
        self,
        model_id: str,
        text: str,
        params: GatewayTTSParams | None,
        api_key: str | None,
        *,
        stream: bool = False,
    ) -> tuple[str, str, bytes, Headers]:
        route, name = model_id.split("/", 1)
        api_key = api_key or self.gateway.token
        options: dict[str, Any] = dict(params or {})

        match route:
            case "openai":
                body = {"model": name, "input": text} | options
                headers = Header.AUTHORIZATION.bearer(api_key) if api_key else {}
                return route, self.gateway.url("openai/audio/speech"), to_json(body), headers
            case "google-ai-studio":
                config: dict[str, Any] = {"responseModalities": ["AUDIO"]}
                if voice := options.get("voice"):
                    config["speechConfig"] = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}

                body = {"contents": [{"parts": [{"text": text}]}], "generationConfig": config}
                headers: Headers = {Header.GOOGLE_API_KEY: api_key} if api_key else {}
                action = "streamGenerateContent?alt=sse" if stream else "generateContent"
                url = self.gateway.url(f"google-ai-studio/v1beta/models/{name}:{action}")
                return route, url, to_json(body), headers
            case _:
                raise ValueError(f"Unknown AI Gateway TTS route: {route!r} in model id {model_id!r}")

    async def run(
        self,
        model_id: str,
        text: str,
        *,
        params: GatewayTTSParams | None = None,
        api_key: str | None = None,
        timeout: Timeout = None,
    ) -> BinaryAudio:
        route, url, content, headers = self._request(model_id, text, params, api_key)

        async with self.gateway.http() as http:
            response = await http.post(
                url,
                content=content,
                headers=CommonHeaders.JSON | headers,
                timeout=resolve_timeout(timeout),
            )
            response.raise_for_status()

        if route == "google-ai-studio":
            result = GeminiResponse.model_validate_json(response.content)
            if audio := result.audio():
                return binary_audio(audio, GEMINI_MEDIA_TYPE)
            raise _no_audio(result.finish_reason())

        return binary_audio(response.content, content_type(response))

    @asynccontextmanager
    async def stream(
        self,
        model_id: str,
        text: str,
        *,
        params: GatewayTTSParams | None = None,
        api_key: str | None = None,
        timeout: Timeout = None,
    ) -> AsyncGenerator[AudioStream]:
        route, url, content, headers = self._request(model_id, text, params, api_key, stream=True)

        async with post_stream(self.gateway, url, content, headers=headers, timeout=timeout) as response:
            if route == "google-ai-studio":
                yield AudioStream(GEMINI_MEDIA_TYPE, self._gemini_chunks(response))
            else:
                yield await AudioStream.start(response.aiter_bytes(), content_type(response))

    async def _gemini_chunks(self, response: httpx2.Response) -> AsyncIterator[bytes]:
        spoke, reason = False, None

        async for event in EventSource(response):
            result = GeminiResponse.model_validate_json(event.data)
            reason = result.finish_reason() or reason
            if audio := result.audio():
                spoke = True
                yield audio

        if not spoke:
            raise _no_audio(reason)
