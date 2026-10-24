from collections.abc import Mapping
from typing import Literal, TypeAlias

from httpx2._types import TimeoutTypes
from pydantic import BaseModel, JsonValue

__all__ = [
    "Headers",
    "JsonObject",
    "RunInput",
    "RunOutput",
    "TTSTransport",
    "Timeout",
]

Headers: TypeAlias = dict[str, str]
Timeout: TypeAlias = TimeoutTypes | None

JsonObject: TypeAlias = Mapping[str, JsonValue]
RunInput: TypeAlias = BaseModel | JsonObject
RunOutput: TypeAlias = JsonValue | bytes

TTSTransport: TypeAlias = Literal["http", "ws"]
