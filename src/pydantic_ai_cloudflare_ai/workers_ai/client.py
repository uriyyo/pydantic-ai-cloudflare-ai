from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import httpx2
from pydantic import BaseModel
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_core import to_json

from pydantic_ai_cloudflare_ai import urls
from pydantic_ai_cloudflare_ai._utils import CloudflareEnvVar, resolve_timeout
from pydantic_ai_cloudflare_ai.base import CloudflareAI
from pydantic_ai_cloudflare_ai.headers import CommonHeaders, Header
from pydantic_ai_cloudflare_ai.tts.workers_ai import TTS
from pydantic_ai_cloudflare_ai.types import Headers

from .chat import WorkersAIChatModel
from .schemas import RunRequest, RunResult

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.models.typesafe import TypeSafeModel

    from pydantic_ai_cloudflare_ai.types import RunInput, RunOutput, Timeout

__all__ = [
    "WorkersAI",
    "workers_ai_embedding_model",
    "workers_ai_model",
]


@dataclass(frozen=True)
class WorkersAI(CloudflareAI):
    account_id: str = field(
        default_factory=lambda: CloudflareEnvVar.CLOUDFLARE_ACCOUNT_ID.lookup("WorkersAI(account_id=...)")
    )
    api_token: str = field(
        default_factory=lambda: CloudflareEnvVar.CLOUDFLARE_API_TOKEN.lookup("WorkersAI(api_token=...)")
    )
    gateway_id: str | None = field(default_factory=CloudflareEnvVar.CLOUDFLARE_GATEWAY_ID.get)
    api_url: str = field(default="https://api.cloudflare.com/client/v4/accounts")

    def __post_init__(self) -> None:
        urls.validate_http(self.api_url)

    def model(self, model_id: str, *, api: Literal["chat", "responses"] = "chat") -> Model:
        # `@cf/...` runs on Cloudflare; other providers' models go through the account's unified endpoints.
        if model_id.startswith("@"):
            return WorkersAIChatModel(model_id, provider=self.provider())

        if model_id.startswith("anthropic/"):
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider

            http = self.http()
            http.event_hooks["request"].append(_drop_query)
            provider = AnthropicProvider(
                base_url=urls.join(self.api_url, self.account_id, "ai"),
                api_key=self.api_token,
                http_client=http,
            )
            return AnthropicModel(model_id, provider=provider)

        if api == "responses":
            from pydantic_ai.models.openai import OpenAIResponsesModel

            return OpenAIResponsesModel(model_id, provider=self.provider())

        from pydantic_ai.models.openai import OpenAIChatModel

        return OpenAIChatModel(model_id, provider=self.provider())

    def embedding_model(self, model_id: str) -> OpenAIEmbeddingModel:
        return OpenAIEmbeddingModel(model_id, provider=self.provider())

    def decision_model(self, model_id: str = "typesafe/jev") -> TypeSafeModel:
        from pydantic_ai.models.typesafe import TypeSafeModel
        from pydantic_ai.providers.typesafe import TypeSafeProvider

        from .decision import RunTransport

        url = self.url("run")
        client = httpx2.AsyncClient(transport=RunTransport(url, model_id), headers=self.headers())
        provider = TypeSafeProvider(
            api_key=self.api_token,
            base_url=url,
            http_client=client,
        )

        return TypeSafeModel(model_id, provider=provider)

    @cached_property
    def tts(self) -> TTS:
        return TTS(self)

    async def run(
        self,
        model_id: str,
        body: RunInput,
        *,
        timeout: Timeout = None,
    ) -> RunOutput:
        payload = body.model_dump(mode="json", exclude_unset=True) if isinstance(body, BaseModel) else body

        # `@cf/` models: model in the path; third-party: model in the body.
        if model_id.startswith("@cf/"):
            url, content = self.url(f"run/{model_id}"), to_json(payload)
        else:
            url, content = self.url("run"), RunRequest(model=model_id, input=payload).model_dump_json()

        async with self.http() as client:
            response = await client.post(
                url, content=content, headers=CommonHeaders.JSON, timeout=resolve_timeout(timeout)
            )
            response.raise_for_status()

            if response.headers.get(Header.CONTENT_TYPE, "").startswith("application/json"):
                return RunResult.model_validate_json(response.content).result

            return response.content

    def provider(self) -> OpenAIProvider:
        return OpenAIProvider(
            base_url=self.url("v1"),
            api_key=self.api_token,
            http_client=self.http(),
        )

    def url(self, path: str) -> str:
        return urls.join(
            self.api_url,
            self.account_id,
            "ai",
            path,
        )

    def headers(self) -> Headers:
        _headers = Header.AUTHORIZATION.bearer(self.api_token)

        if self.gateway_id:
            _headers[Header.GATEWAY_ID] = self.gateway_id
        if self.options:
            _headers.update(self.options.headers())

        return _headers


async def _drop_query(request: httpx2.Request) -> None:
    request.url = request.url.copy_with(query=None)


def workers_ai_model(model_id: str) -> Model:
    return WorkersAI().model(model_id)


def workers_ai_embedding_model(model_id: str) -> OpenAIEmbeddingModel:
    return WorkersAI().embedding_model(model_id)
