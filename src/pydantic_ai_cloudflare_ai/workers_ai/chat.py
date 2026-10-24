from collections.abc import Sequence
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings

__all__ = [
    "WorkersAIChatModel",
    "flatten_content",
]


def flatten_content(message: ChatCompletionMessageParam) -> ChatCompletionMessageParam:
    content: Any = message.get("content")

    if isinstance(content, str):
        return message

    texts: list[str] = []
    for part in content or ():
        match part:
            case {"type": "text", "text": text} | {"type": "refusal", "refusal": text}:
                texts.append(text)
            case _:
                raise UserError(
                    f"Workers AI text models accept text content only, got a {part.get('type')!r} content part."
                )

    return cast(
        "ChatCompletionMessageParam",
        {**message, "content": "\n\n".join(texts)},
    )


class WorkersAIChatModel(OpenAIChatModel):
    async def _map_messages(
        self,
        messages: Sequence[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        *,
        model_settings: ModelSettings | None = None,
    ) -> list[ChatCompletionMessageParam]:
        mapped = await super()._map_messages(
            messages,
            model_request_parameters,
            model_settings=model_settings,
        )
        return [flatten_content(message) for message in mapped]
