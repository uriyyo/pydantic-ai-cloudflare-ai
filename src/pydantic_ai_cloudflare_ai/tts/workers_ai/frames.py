from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Discriminator, JsonValue, Tag, TypeAdapter

from pydantic_ai_cloudflare_ai.tts.errors import TTSError

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "GrokFrames",
    "TTSError",
    "TTSFrames",
    "frames_for",
]


class TTSFrames(Protocol):
    media_type: str | None

    def send(self, text: str) -> Iterable[str]:
        pass

    def receive(self, message: str | bytes) -> bytes | None:
        pass


class TextDelta(BaseModel):
    type: Literal["text.delta"] = "text.delta"
    delta: str


class TextDone(BaseModel):
    type: Literal["text.done"] = "text.done"


class AudioDelta(BaseModel):
    type: Literal["audio.delta"]
    delta: str


class AudioDone(BaseModel):
    type: Literal["audio.done"]


class ErrorFrame(BaseModel):
    type: Literal["error"]
    message: str | None = None


class OtherFrame(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str


def _tag(frame: JsonValue) -> str:
    match frame:
        case {"type": str() as kind} if kind in {"audio.delta", "audio.done", "error"}:
            return kind
        case _:
            return "other"


ServerFrame: TypeAlias = Annotated[
    Annotated[AudioDelta, Tag("audio.delta")]
    | Annotated[AudioDone, Tag("audio.done")]
    | Annotated[ErrorFrame, Tag("error")]
    | Annotated[OtherFrame, Tag("other")],
    Discriminator(_tag),
]

server_frame = TypeAdapter[ServerFrame](ServerFrame)


class GrokFrames:
    media_type: str | None = "audio/mpeg"

    def send(self, text: str) -> Iterable[str]:
        return (TextDelta(delta=text).model_dump_json(), TextDone().model_dump_json())

    def receive(self, message: str | bytes) -> bytes | None:
        match server_frame.validate_json(message):
            case AudioDelta(delta=delta):
                return base64.b64decode(delta)
            case AudioDone():
                return None
            case ErrorFrame(message=error):
                raise TTSError(error)
            case _:
                return b""


def frames_for(model_id: str) -> TTSFrames:
    match model_id:
        case "xai/grok-tts":
            return GrokFrames()
        case _:
            raise ValueError(f"No websocket frames known for {model_id!r}; pass `frames=` to `tts.stream`")
