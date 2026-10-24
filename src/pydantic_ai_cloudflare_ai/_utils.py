from __future__ import annotations

import os
from enum import StrEnum, unique
from typing import TYPE_CHECKING

import httpx2
from pydantic_ai.models import create_async_httpx2_client
from pydantic_ai.providers import missing_api_key_error

from .types import Headers

if TYPE_CHECKING:
    from httpx2._client import UseClientDefault
    from httpx2._types import TimeoutTypes

    from .types import Timeout

__all__ = [
    "CloudflareEnvVar",
    "http_client",
]


@unique
class CloudflareEnvVar(StrEnum):
    CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
    CLOUDFLARE_API_TOKEN = "CLOUDFLARE_API_TOKEN"  # noqa: S105
    CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"
    CLOUDFLARE_GATEWAY_TOKEN = "CLOUDFLARE_GATEWAY_TOKEN"  # noqa: S105

    def lookup(self, owner: str) -> str:
        if value := self.get():
            return value

        raise missing_api_key_error(f"Set the `{self}` environment variable or pass it via `{owner}`")

    def get(self) -> str | None:
        return os.getenv(self)


class _ThroughClient(httpx2.AsyncBaseTransport):
    def __init__(self, client: httpx2.AsyncClient) -> None:
        self.client = client

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        return await self.client.send(request, stream=True)

    async def aclose(self) -> None:
        pass


def http_client(
    client: httpx2.AsyncClient | None,
    headers: Headers,
) -> httpx2.AsyncClient:
    if client is None:
        client = create_async_httpx2_client()
        client.headers.update(headers)
        return client

    return httpx2.AsyncClient(
        headers={**client.headers, **headers},
        timeout=client.timeout,
        transport=_ThroughClient(client),
    )


def resolve_timeout(timeout: Timeout) -> TimeoutTypes | UseClientDefault:
    return httpx2.USE_CLIENT_DEFAULT if timeout is None else timeout


def read_timeout(timeout: Timeout) -> float | None:
    return httpx2.Timeout(timeout).read if timeout is not None else None
