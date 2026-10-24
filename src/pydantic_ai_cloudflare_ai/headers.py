from __future__ import annotations

from enum import Enum, StrEnum, unique
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Headers

__all__ = [
    "CommonHeaders",
    "Header",
]


@unique
class Header(StrEnum):
    AUTHORIZATION = "Authorization"
    CONTENT_TYPE = "content-type"
    CONTENT_LENGTH = "content-length"
    GATEWAY_ID = "cf-aig-gateway-id"
    GATEWAY_AUTHORIZATION = "cf-aig-authorization"
    GOOGLE_API_KEY = "x-goog-api-key"

    def bearer(self, token: str) -> Headers:
        return {self: f"Bearer {token}"}


class CommonHeaders(dict[str, str], Enum):
    JSON = {Header.CONTENT_TYPE: "application/json"}  # noqa: RUF012
