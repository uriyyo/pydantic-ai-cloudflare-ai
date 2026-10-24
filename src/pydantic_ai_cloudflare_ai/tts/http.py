from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from pydantic_ai.models import create_async_httpx2_client

from pydantic_ai_cloudflare_ai._utils import resolve_timeout
from pydantic_ai_cloudflare_ai.headers import CommonHeaders, Header

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import httpx2

    from pydantic_ai_cloudflare_ai.base import CloudflareAI
    from pydantic_ai_cloudflare_ai.types import Headers, Timeout

__all__ = [
    "content_type",
    "download",
    "post_stream",
]


def content_type(response: httpx2.Response) -> str | None:
    return response.headers.get(Header.CONTENT_TYPE)


async def download(url: str, *, timeout: Timeout = None) -> httpx2.Response:
    async with create_async_httpx2_client() as http:
        response = await http.get(url, timeout=resolve_timeout(timeout))
        response.raise_for_status()

        return response


@asynccontextmanager
async def post_stream(
    ai: CloudflareAI,
    url: str,
    content: str | bytes,
    *,
    headers: Headers | None = None,
    timeout: Timeout = None,
) -> AsyncGenerator[httpx2.Response]:
    async with (
        ai.http() as http,
        http.stream(
            "POST",
            url,
            content=content,
            headers=CommonHeaders.JSON | (headers or {}),
            timeout=resolve_timeout(timeout),
        ) as response,
    ):
        response.raise_for_status()

        yield response
