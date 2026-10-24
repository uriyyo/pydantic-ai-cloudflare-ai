import pytest
import respx

from pydantic_ai_cloudflare_ai import WorkersAI
from pydantic_ai_cloudflare_ai.tts import AudioStream, media_type


@pytest.mark.parametrize(
    ("data", "declared", "expected"),
    [
        (b"anything", "audio/L16;rate=24000", "audio/L16;rate=24000"),
        (b"RIFF\0\0\0\0WAVEfmt ", "application/octet-stream", "audio/wav"),
        (b"OggS\0", None, "audio/ogg"),
        (b"fLaC\0", None, "audio/flac"),
        (b"ID3\4", None, "audio/mpeg"),
        (b"\xff\xfb\x90", None, "audio/mpeg"),
        (b"\xff\xf1\x50", None, "audio/aac"),
        (b"", None, "audio/*"),
        (b"???", "binary/octet-stream", "audio/*"),
    ],
)
def test_media_type(data, declared, expected):
    assert media_type(data, declared) == expected


async def test_stream_without_audio_type_tells_format_from_first_chunk():
    async def chunks():
        yield b"OggS-1"
        yield b"-2"

    stream = await AudioStream.start(chunks(), "application/octet-stream")

    assert stream.media_type == "audio/ogg"
    assert [c async for c in stream] == [b"OggS-1", b"-2"]


async def test_untyped_http_stream_gets_audio_media_type(httpx2_mock: respx.Router):
    httpx2_mock.post("https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/deepgram/aura-2-en").respond(
        content=b"ID3...",
        headers={"content-type": "application/octet-stream"},
    )
    wai = WorkersAI(account_id="acct", api_token="t", gateway_id=None)

    async with wai.tts.stream("@cf/deepgram/aura-2-en", "hi") as audio:
        whole = await audio.read()

    assert whole.media_type == "audio/mpeg"
    assert whole.data == b"ID3..."
