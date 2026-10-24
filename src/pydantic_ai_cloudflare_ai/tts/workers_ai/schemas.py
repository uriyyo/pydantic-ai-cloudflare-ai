from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

from pydantic import BaseModel, ConfigDict, JsonValue

from pydantic_ai_cloudflare_ai.types import JsonObject

__all__ = [
    "AudioResponse",
    "AuraParams",
    "GeminiParams",
    "GrokParams",
    "InworldParams",
    "MeloParams",
    "MiniMaxParams",
    "OpenAIParams",
    "TTSParams",
    "TTSRequest",
    "WSQuery",
]


class AuraParams(TypedDict, total=False):
    speaker: str
    encoding: str
    container: str
    sample_rate: int
    bit_rate: int


class MeloParams(TypedDict, total=False):
    lang: str


class GrokParams(TypedDict, total=False):
    language: str
    voice_id: str
    speed: float
    output_format: dict[str, JsonValue]
    text_normalization: bool


class GeminiParams(TypedDict, total=False):
    voice: str
    temperature: float


class OpenAIParams(TypedDict, total=False):
    voice: str
    response_format: str
    speed: float


class MiniMaxParams(TypedDict, total=False):
    voice_id: str
    speed: float
    volume: float
    pitch: int
    emotion: str
    format: str
    sample_rate: int


class InworldParams(TypedDict, total=False):
    voice_id: str
    output_format: str
    bit_rate: int
    sample_rate: int
    speaking_rate: float
    temperature: float


TTSParams: TypeAlias = (
    AuraParams | MeloParams | GrokParams | GeminiParams | OpenAIParams | MiniMaxParams | InworldParams | JsonObject
)


class TTSRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str | None = None
    prompt: str | None = None


class WSQuery(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    websocket: Literal["true"] = "true"


class AudioResponse(BaseModel):
    audio: str
