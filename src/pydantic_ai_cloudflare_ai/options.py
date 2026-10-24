from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from annotated_types import Ge, Gt, Le, MaxLen
from pydantic import AfterValidator, BaseModel, ConfigDict
from pydantic_core import to_json

from .types import Headers

__all__ = [
    "CustomCost",
    "GatewayOptions",
]

MetadataValue: TypeAlias = str | int | float | bool


def _validate_not_reserved(key: str) -> str:
    if key.startswith("cf."):
        raise ValueError("`cf.*` metadata keys are reserved by Cloudflare")

    return key


MetadataKey: TypeAlias = Annotated[
    str,
    AfterValidator(_validate_not_reserved),
]


class CustomCost(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    per_token_in: float
    per_token_out: float
    per_cache_read_token: float | None = None
    per_cache_write_token: float | None = None


def _dump_value(value: Any) -> str:
    match value:
        case str():
            return value
        case _:
            return to_json(value).decode()


class GatewayOptions(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    cache_ttl: Annotated[int, Ge(60)] | None = None
    cache_key: str | None = None
    skip_cache: bool | None = None
    metadata: Annotated[dict[MetadataKey, MetadataValue], MaxLen(5)] | None = None
    collect_log: bool | None = None
    collect_log_payload: bool | None = None
    request_timeout: Annotated[int, Gt(0)] | None = None
    max_attempts: Annotated[int, Ge(1), Le(5)] | None = None
    retry_delay: Annotated[int, Ge(0), Le(60_000)] | None = None
    backoff: Literal["constant", "linear", "exponential"] | None = None
    custom_cost: CustomCost | None = None
    no_wholesale: bool | None = None
    byok_alias: str | None = None

    def headers(self) -> Headers:
        # Every option is the `cf-aig-` header of the same name; strings go as is, the rest as JSON.
        return {
            f"cf-aig-{name.replace('_', '-')}": _dump_value(value)
            for name, value in self.model_dump(exclude_none=True).items()
        }
