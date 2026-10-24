from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import httpx
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_ai_cloudflare_ai import urls
from pydantic_ai_cloudflare_ai._utils import CloudflareEnvVar
from pydantic_ai_cloudflare_ai.base import CloudflareAI
from pydantic_ai_cloudflare_ai.headers import Header
from pydantic_ai_cloudflare_ai.types import Headers

from .tts import GatewayTTS

if TYPE_CHECKING:
    from pydantic_ai.embeddings import EmbeddingModel
    from pydantic_ai.models import Model

__all__ = [
    "AIGateway",
    "gateway_embedding_model",
    "gateway_model",
]


@dataclass(frozen=True)
class AIGateway(CloudflareAI):
    account_id: str = field(
        default_factory=lambda: CloudflareEnvVar.CLOUDFLARE_ACCOUNT_ID.lookup(
            "AIGateway(account_id=...)",
        )
    )
    # Cloudflare creates the `default` gateway, with authentication on, on the first authenticated request.
    gateway_id: str = field(default_factory=lambda: CloudflareEnvVar.CLOUDFLARE_GATEWAY_ID.get() or "default")
    token: str | None = field(default_factory=CloudflareEnvVar.CLOUDFLARE_GATEWAY_TOKEN.get)

    api_url: str = field(default="https://gateway.ai.cloudflare.com/v1")

    def __post_init__(self) -> None:
        urls.validate_http(self.api_url)

    def model(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
    ) -> Model:
        route, name = model_id.split("/", 1)
        api_key = api_key or self.token

        match route:
            case "openai":
                from pydantic_ai.models.openai import OpenAIResponsesModel

                provider = OpenAIProvider(
                    base_url=self.url(route),
                    api_key=api_key,
                    http_client=self.http(),
                )
                return OpenAIResponsesModel(name, provider=provider)
            case "anthropic":
                from pydantic_ai.models.anthropic import AnthropicModel
                from pydantic_ai.providers.anthropic import AnthropicProvider

                provider = AnthropicProvider(
                    base_url=self.url(route),
                    api_key=api_key,
                    http_client=self.http(),
                )
                return AnthropicModel(name, provider=provider)
            case "google-ai-studio":
                from pydantic_ai.models.google import GoogleModel
                from pydantic_ai.providers.google import GoogleProvider

                provider = GoogleProvider(
                    base_url=self.url(route),
                    api_key=api_key,
                    http_client=self.http(),
                )
                return GoogleModel(name, provider=provider)
            case "groq":
                from pydantic_ai.models.groq import GroqModel
                from pydantic_ai.providers.groq import GroqProvider

                # The Groq SDK still takes a legacy `httpx` client.
                client = httpx.AsyncClient(headers=self.headers())
                provider = GroqProvider(
                    base_url=self.url(route),
                    api_key=api_key,
                    http_client=client,
                )

                return GroqModel(name, provider=provider)
            case "mistral":
                from pydantic_ai.models.mistral import MistralModel
                from pydantic_ai.providers.mistral import MistralProvider

                provider = MistralProvider(
                    base_url=self.url(route),
                    api_key=api_key,
                    http_client=self.http(),
                )
                return MistralModel(name, provider=provider)
            case _:
                from pydantic_ai.models.openai import OpenAIChatModel
                from pydantic_ai.profiles import ModelProfile

                # Any other route is a gateway custom provider, served OpenAI-compatible at `custom-{slug}`.
                route = route if route.startswith("custom-") else f"custom-{route}"
                provider = OpenAIProvider(
                    base_url=self.url(route),
                    api_key=api_key,
                    http_client=self.http(),
                )
                # Unknown model names get a non-thinking profile, which silently drops `thinking`; send it as
                # `reasoning_effort` and let the provider reject it if unsupported.
                return OpenAIChatModel(name, provider=provider, profile=ModelProfile(supports_thinking=True))

    def embedding_model(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
    ) -> EmbeddingModel:
        route, name = model_id.split("/", 1)
        api_key = api_key or self.token

        if route != "openai":
            raise ValueError(f"Unknown AI Gateway embedding route: {route!r} in model id {model_id!r}")

        provider = OpenAIProvider(
            base_url=self.url(route),
            api_key=api_key,
            http_client=self.http(),
        )

        return OpenAIEmbeddingModel(
            name,
            provider=provider,
        )

    @cached_property
    def tts(self) -> GatewayTTS:
        return GatewayTTS(self)

    def url(self, path: str) -> str:
        return urls.join(
            self.api_url,
            self.account_id,
            self.gateway_id,
            path,
        )

    def headers(self) -> Headers:
        headers = {}

        if self.token:
            headers.update(Header.GATEWAY_AUTHORIZATION.bearer(self.token))
        if self.options:
            headers.update(self.options.headers())

        return headers


def gateway_model(model_id: str, *, api_key: str | None = None) -> Model:
    return AIGateway().model(model_id, api_key=api_key)


def gateway_embedding_model(model_id: str, *, api_key: str | None = None) -> EmbeddingModel:
    return AIGateway().embedding_model(model_id, api_key=api_key)
