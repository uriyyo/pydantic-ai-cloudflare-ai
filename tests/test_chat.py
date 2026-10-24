import pytest
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from pydantic_ai_cloudflare_ai import WorkersAI

MODEL = WorkersAI(account_id="acct", api_token="cf-token").model("@cf/openai/gpt-oss-120b")


async def test_message_content_is_flattened_to_strings():
    messages = [
        ModelRequest(parts=[UserPromptPart(content=["first chunk", "second chunk"])]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="edit",
                    args={"a": 1},
                    tool_call_id="call-1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="edit",
                    content="done",
                    tool_call_id="call-1",
                )
            ]
        ),
    ]
    mapped = await MODEL._map_messages(messages, ModelRequestParameters())  # noqa: SLF001
    assert [message["content"] for message in mapped] == ["first chunk\n\nsecond chunk", "", "done"]


async def test_non_text_content_is_rejected():
    messages = [
        ModelRequest(parts=[UserPromptPart(content=["look", BinaryContent(b"\x89PNG", media_type="image/png")])])
    ]
    with pytest.raises(UserError, match="text content only"):
        await MODEL._map_messages(messages, ModelRequestParameters())  # noqa: SLF001
