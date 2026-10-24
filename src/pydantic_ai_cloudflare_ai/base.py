from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._utils import http_client
from .types import Headers

if TYPE_CHECKING:
    import httpx2
    from pydantic_ai.embeddings import EmbeddingModel
    from pydantic_ai.models import Model

    from .options import GatewayOptions

__all__ = [
    "CloudflareAI",
]


@dataclass(frozen=True)
class CloudflareAI(ABC):
    account_id: str
    http_client: httpx2.AsyncClient | None = None
    options: GatewayOptions | None = None

    @abstractmethod
    def model(self, model_id: str) -> Model: ...

    @abstractmethod
    def embedding_model(self, model_id: str) -> EmbeddingModel: ...

    @abstractmethod
    def url(self, path: str) -> str: ...

    @abstractmethod
    def headers(self) -> Headers: ...

    def http(self) -> httpx2.AsyncClient:
        return http_client(self.http_client, self.headers())
