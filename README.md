# pydantic-ai-cloudflare-ai

[Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) and
[AI Gateway](https://developers.cloudflare.com/ai-gateway/) for [Pydantic AI](https://ai.pydantic.dev):
chat, embeddings, text-to-speech and gateway options.

```bash
pip install pydantic-ai-cloudflare-ai
pip install "pydantic-ai-cloudflare-ai[all]"  # + anthropic, google, groq, mistral, typesafe, ws
```

Configured from `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_GATEWAY_ID` and
`CLOUDFLARE_GATEWAY_TOKEN`, or the same values passed explicitly.

## Workers AI

Cloudflare-hosted `@cf/...` models, and other providers' models through your account's unified endpoints:
one Cloudflare API token, Cloudflare billing, no provider keys.

```python
from pydantic_ai import Agent, Embedder
from pydantic_ai_cloudflare_ai import WorkersAI

wai = WorkersAI(gateway_id="default")  # gateway_id is optional: logs requests through AI Gateway

agent = Agent(wai.model("@cf/meta/llama-3.3-70b-instruct-fp8-fast"))
agent = Agent(wai.model("openai/gpt-5-mini"))  # OpenAI Chat Completions
agent = Agent(wai.model("openai/gpt-5-mini", api="responses"))  # OpenAI Responses
agent = Agent(wai.model("anthropic/claude-haiku-4.5"))  # Anthropic Messages
agent = Agent(wai.model("google/gemini-2.5-flash"))
embedder = Embedder(wai.embedding_model("@cf/baai/bge-m3"))
result = await wai.run("@cf/openai/whisper", {...})  # any other model
```

Model ids are Cloudflare's catalog ids (`google/...`, `anthropic/claude-haiku-4.5`), not the providers' own.

## AI Gateway

Provider-native routes, for your own provider keys (or keys stored in the gateway).

```python
from pydantic_ai_cloudflare_ai import AIGateway

gw = AIGateway()

agent = Agent(gw.model("openai/gpt-5.6"))
agent = Agent(gw.model("anthropic/claude-sonnet-4-5", api_key="sk-ant-..."))
embedder = Embedder(gw.embedding_model("openai/text-embedding-3-small"))
```

Routes: `openai`, `anthropic`, `google-ai-studio`, `groq`, `mistral`. Any other route is a gateway
[custom provider](https://developers.cloudflare.com/ai-gateway/configuration/custom-providers/), called as
OpenAI-compatible chat: `gw.model("xiaomi/mimo2.6-flash")` (or `"custom-xiaomi/..."`). The provider key is `api_key`, else the
gateway token (BYOK / unified billing), else the provider's own env var. Without a gateway id it uses
Cloudflare's `default` gateway, created on first use with authentication on, so it needs `CLOUDFLARE_GATEWAY_TOKEN`.

## Text-to-speech

Every Workers AI catalog voice (`@cf/...`, `openai/...`, `google/...`, `elevenlabs/...`, `xai/...`, ...),
plus OpenAI and Gemini voices through the gateway. Results are Pydantic AI `BinaryAudio`.

```python
audio = await wai.tts.run("openai/tts-1", "Hello", params={"voice": "nova"})
audio.data, audio.media_type  # b"...", "audio/mpeg"

async with wai.tts.stream("@cf/deepgram/aura-1", "Hello") as stream:
    stream.media_type  # known before the first chunk
    async for chunk in stream:
        ...

async with wai.tts.stream("xai/grok-tts", "Hello", transport="ws") as stream:  # `ws` extra
    audio = await stream.read()

audio = await gw.tts.run("google-ai-studio/gemini-2.5-flash-preview-tts", "Hello", params={"voice": "Kore"})
```

## Gateway options

Caching, retries, timeouts, logging, metadata and custom costs, validated and sent as `cf-aig-*` headers.

```python
from pydantic_ai_cloudflare_ai import GatewayOptions

gw = AIGateway(options=GatewayOptions(cache_ttl=3600, max_attempts=3, backoff="exponential"))

await agent.run("...", model_settings={"extra_headers": GatewayOptions(skip_cache=True).headers()})  # one request
```

## Model strings

```python
import pydantic_ai_cloudflare_ai

pydantic_ai_cloudflare_ai.install()

agent = Agent("cloudflare/workers-ai:@cf/meta/llama-3.3-70b-instruct-fp8-fast")
agent = Agent("cloudflare/openai:gpt-5.6")
```
