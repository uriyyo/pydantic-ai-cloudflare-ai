from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from pydantic_ai.messages import BinaryAudio

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

__all__ = [
    "AudioStream",
    "binary_audio",
    "media_type",
]

# Audio whose format neither the provider nor its first bytes give away.
UNKNOWN_AUDIO = "audio/*"


def _frame_sync(data: bytes, mask: int, value: int) -> bool:
    return len(data) >= 2 and data[0] == 0xFF and data[1] & mask == value


SIGNATURES: tuple[tuple[Callable[[bytes], bool], str], ...] = (
    (lambda data: data[:4] == b"RIFF" and data[8:12] == b"WAVE", "audio/wav"),
    (lambda data: data.startswith(b"OggS"), "audio/ogg"),
    (lambda data: data.startswith(b"fLaC"), "audio/flac"),
    (lambda data: data.startswith(b"ID3"), "audio/mpeg"),
    # ADTS AAC shares the MPEG frame sync, so it is checked first.
    (lambda data: _frame_sync(data, 0xF6, 0xF0), "audio/aac"),
    (lambda data: _frame_sync(data, 0xE0, 0xE0), "audio/mpeg"),
)


def media_type(data: bytes, declared: str | None = None) -> str:
    """`declared` when it is an audio type, else the format the first bytes of `data` show."""
    if declared and declared.startswith("audio/"):
        return declared

    return next((kind for matches, kind in SIGNATURES if matches(data)), UNKNOWN_AUDIO)


def binary_audio(data: bytes, declared: str | None = None) -> BinaryAudio:
    return BinaryAudio(data, media_type=media_type(data, declared))


@dataclass(frozen=True)
class AudioStream:
    """Audio as it is spoken: `media_type` up front, `async for chunk in stream` for the bytes."""

    media_type: str
    chunks: AsyncIterator[bytes]

    @classmethod
    def of(cls, whole: BinaryAudio) -> Self:
        async def chunks() -> AsyncIterator[bytes]:
            yield whole.data

        return cls(whole.media_type, chunks())

    @classmethod
    async def start(cls, chunks: AsyncIterator[bytes], declared: str | None) -> Self:
        if declared and declared.startswith("audio/"):
            return cls(declared, chunks)

        # No audio type declared: read the first chunk to tell the format, then hand it out first.
        first = await anext(chunks, b"")

        async def replay() -> AsyncIterator[bytes]:
            if first:
                yield first
            async for chunk in chunks:
                yield chunk

        return cls(media_type(first), replay())

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self.chunks

    async def read(self) -> BinaryAudio:
        """The rest of the stream as one piece of audio."""
        data = b"".join([chunk async for chunk in self])
        return BinaryAudio(data, media_type=self.media_type)
