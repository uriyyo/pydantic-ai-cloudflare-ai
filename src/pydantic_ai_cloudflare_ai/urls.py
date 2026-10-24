from __future__ import annotations

from pydantic import AnyHttpUrl, TypeAdapter
from pydantic_core import Url

__all__ = [
    "join",
    "validate_http",
    "websocket",
]

http_url = TypeAdapter[AnyHttpUrl](AnyHttpUrl)

WS_SCHEMES = {
    "https": "wss",
    "http": "ws",
}


def validate_http(url: str) -> str:
    return str(http_url.validate_python(url)).rstrip("/")


def join(base: str, *parts: str) -> str:
    return "/".join((base.rstrip("/"), *parts))


def websocket(url: str) -> str:
    parsed = Url(url)

    return str(
        Url.build(
            scheme=WS_SCHEMES[parsed.scheme],
            host=parsed.host or "",
            port=parsed.port,
            path=(parsed.path or "").lstrip("/"),
            query=parsed.query,
        )
    )
