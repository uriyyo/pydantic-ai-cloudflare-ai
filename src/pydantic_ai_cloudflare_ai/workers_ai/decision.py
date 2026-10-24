from __future__ import annotations

from dataclasses import dataclass

import httpx2
from pydantic import BaseModel, ConfigDict
from pydantic_core import to_json

from pydantic_ai_cloudflare_ai.headers import CommonHeaders, Header

from .schemas import RunRequest, RunResult

__all__ = [
    "RunTransport",
]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None


@dataclass
class RunTransport(httpx2.AsyncBaseTransport):
    url: str
    model: str
    inner: httpx2.AsyncBaseTransport

    def __init__(
        self,
        url: str,
        model: str,
        inner: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = url
        self.model = model
        self.inner = inner or httpx2.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        raw = await request.aread()
        body = SystemOneRequest.model_validate_json(raw)
        headers = httpx2.Headers(request.headers)

        del headers[Header.CONTENT_LENGTH]

        run = httpx2.Request(
            "POST",
            self.url,
            headers=headers,
            content=RunRequest(model=self.model, input=body.model_dump(exclude={"model"})).model_dump_json(),
            extensions=request.extensions,
        )

        response = await self.inner.handle_async_request(run)
        await response.aread()

        if response.status_code != httpx2.codes.OK:
            return response

        # Third-party results are wrapped twice.
        match RunResult.model_validate_json(response.content).result:
            case {"result": inner}:
                result = inner
            case outer:
                result = outer

        return httpx2.Response(
            httpx2.codes.OK,
            content=to_json(result),
            headers=CommonHeaders.JSON,
        )

    async def aclose(self) -> None:
        await self.inner.aclose()
