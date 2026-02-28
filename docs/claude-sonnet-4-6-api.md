# Claude Sonnet 4.6 API Reference

Model ID: `claude-sonnet-4-6`
Released: Feb 17, 2026

## Key Specs
- Context window: 200K tokens (1M in beta)
- Max output tokens: 64K
- API version header: `anthropic-version: 2023-06-01`

## Breaking Changes from Older Models

1. **No assistant prefill** — returns 400. Use structured outputs or append a user message.
2. **Temperature + top_p can't both be specified** — use one or the other.
3. **`thinking` is a direct SDK parameter** — not via `extra_body`.
4. **Streaming required for large max_tokens** — use `stream()` + `get_final_message()`.

## Extended Thinking

```python
# Manual mode (budget_tokens still supported on Sonnet 4.6)
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[...]
)

# Adaptive mode (model decides when/how much to think)
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={"type": "adaptive"},
    messages=[...]
)
```

## Interleaved Thinking + Tools

Requires beta header:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    tools=[...],
    messages=[...],
    extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"}
)
```

When using interleaved thinking with tools, `budget_tokens` can exceed `max_tokens` (up to 200K context window).

## Streaming for Large Requests

```python
# Use stream() + get_final_message() to avoid SDK timeout
async with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=64000,
    messages=[...]
) as stream:
    response = await stream.get_final_message()
```

## Prompt Caching

Supported. Minimum 1024 tokens. Ephemeral type with 5-minute default lifetime.

```python
system=[{
    "type": "text",
    "text": "...",
    "cache_control": {"type": "ephemeral"}
}]
```

## Pricing (USD per MTok)
- Base input: $3.00
- Cache writes (5m): $3.75
- Cache hits: $0.30
- Output: $15.00
