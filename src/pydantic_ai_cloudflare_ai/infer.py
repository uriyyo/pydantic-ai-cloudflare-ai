from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

from .gateway import gateway_embedding_model, gateway_model
from .workers_ai import WorkersAI, workers_ai_embedding_model, workers_ai_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_ai.embeddings import EmbeddingModel, KnownEmbeddingModelName
    from pydantic_ai.models import KnownModelName, Model
    from pydantic_ai.providers import Provider

__all__ = [
    "install",
]

_PREFIX = "cloudflare/"


def _route(model: Any) -> tuple[str, str] | None:
    match model:
        case str() if model.startswith(_PREFIX):
            route, _, name = model.removeprefix(_PREFIX).partition(":")
            return route, name
        case _:
            return None


def cloudflare_model(model: Any) -> Model | None:
    match _route(model):
        case None:
            return None
        case ("workers-ai", name) if name.startswith("typesafe/"):
            return WorkersAI().decision_model(name)
        case ("workers-ai", name):
            return workers_ai_model(name)
        case (route, name):
            return gateway_model(f"{route}/{name}")


def cloudflare_embedding_model(model: Any) -> EmbeddingModel | None:
    match _route(model):
        case None:
            return None
        case ("workers-ai", name):
            return workers_ai_embedding_model(name)
        case (route, name):
            return gateway_embedding_model(f"{route}/{name}")


@cache
def install() -> None:

    from pydantic_ai import embeddings, models
    from pydantic_ai.providers import infer_provider

    original_infer_model = models.infer_model
    original_infer_embedding_model = embeddings.infer_embedding_model

    def infer_model(
        model: Model | KnownModelName | str, provider_factory: Callable[[str], Provider[Any]] = infer_provider
    ) -> Model:
        return cloudflare_model(model) or original_infer_model(model, provider_factory)

    def infer_embedding_model(
        model: EmbeddingModel | KnownEmbeddingModelName | str,
        *,
        provider_factory: Callable[[str], Provider[Any]] = infer_provider,
    ) -> EmbeddingModel:
        return cloudflare_embedding_model(model) or original_infer_embedding_model(
            model, provider_factory=provider_factory
        )

    models.infer_model = infer_model  # ty: ignore[invalid-assignment]
    embeddings.infer_embedding_model = infer_embedding_model  # ty: ignore[invalid-assignment]
