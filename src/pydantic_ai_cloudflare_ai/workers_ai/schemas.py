from __future__ import annotations

from pydantic import BaseModel, ConfigDict, JsonValue

__all__ = [
    "RunRequest",
    "RunResult",
]


class RunRequest(BaseModel):
    model: str
    input: JsonValue


class RunResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    result: JsonValue
    success: bool = True
