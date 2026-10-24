"""TTS plumbing shared by `WorkersAI.tts` and `AIGateway.tts`."""

from .audio import AudioStream, binary_audio, media_type
from .errors import TTSError
from .http import content_type, download, post_stream

__all__ = [
    "AudioStream",
    "TTSError",
    "binary_audio",
    "content_type",
    "download",
    "media_type",
    "post_stream",
]
