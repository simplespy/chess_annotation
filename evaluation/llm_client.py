"""
LLM API client with unified interface for OpenAI, Anthropic, and Qwen.

Supports:
- OpenAI GPT models (GPT-4o, GPT-4.1, etc.)
- Anthropic Claude models (Sonnet, Opus, Haiku)
- Qwen models via OpenAI-compatible API (hosted locally or remotely)
- Multi-round tool calling
"""

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class ToolCall:
    """Represents a tool call from an LLM."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Unified LLM response format."""
    text: str
    tool_calls: List[ToolCall]


def call_openai(messages, tools=None, model="gpt-4o", temperature=0.3, base_url=None, api_key=None):
    """
    Call OpenAI API or OpenAI-compatible API with unified interface.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tools: Optional list of tool schemas in OpenAI format
        model: Model name
        temperature: Sampling temperature
        base_url: Optional custom base URL for OpenAI-compatible APIs (e.g., vLLM, Qwen)
        api_key: Optional API key (or use OPENAI_API_KEY env var)

    Returns:
        Tuple of (LLMResponse, raw_message)
    """
    import openai

    kwargs = {}
    if base_url:
        kwargs['base_url'] = base_url
    if api_key:
        kwargs['api_key'] = api_key
    elif not os.getenv('OPENAI_API_KEY'):
        kwargs['api_key'] = 'not-needed'  # For local servers

    client = openai.OpenAI(**kwargs)

    token_key = "max_completion_tokens" if any(m in model for m in ("gpt-4.1", "gpt-5", "o3", "o4")) else "max_tokens"
    req_kwargs = dict(model=model, messages=messages, temperature=temperature, **{token_key: 2048})

    if tools:
        req_kwargs["tools"] = tools
        req_kwargs["tool_choice"] = "auto"

    resp = client.chat.completions.create(**req_kwargs)
    msg = resp.choices[0].message
    text = msg.content or ""
    tc = []
    if msg.tool_calls:
        for c in msg.tool_calls:
            tc.append(ToolCall(id=c.id, name=c.function.name,
                               arguments=json.loads(c.function.arguments)))
    return LLMResponse(text=text, tool_calls=tc), msg


def call_anthropic(messages, tools=None, model="claude-sonnet-4-5-20250929",
                   system=None, temperature=0.3):
    """
    Call Anthropic API with unified interface.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tools: Optional list of tool schemas in Anthropic format
        model: Model name
        system: System message (extracted from messages if not provided)
        temperature: Sampling temperature

    Returns:
        Tuple of (LLMResponse, raw_response)
    """
    import anthropic
    client = anthropic.Anthropic()
    kwargs = dict(model=model, max_tokens=2048, temperature=temperature)
    if system:
        kwargs["system"] = system
    api_messages = [m for m in messages if m["role"] != "system"]
    kwargs["messages"] = api_messages
    if tools:
        kwargs["tools"] = tools
    resp = client.messages.create(**kwargs)
    text = ""
    tc = []
    for block in resp.content:
        if block.type == "text":
            text += block.text
        elif block.type == "tool_use":
            tc.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
    return LLMResponse(text=text, tool_calls=tc), resp


def call_llm(messages, tools=None, provider="openai", model=None, temperature=0.3,
             base_url=None, api_key=None):
    """
    Unified LLM calling interface supporting multiple providers.

    Args:
        messages: List of message dicts
        tools: Tool schemas (provider-specific format)
        provider: 'openai', 'anthropic', or 'qwen'
        model: Model name (provider-specific default if None)
        temperature: Sampling temperature
        base_url: Base URL for OpenAI-compatible APIs (for Qwen)
        api_key: API key (optional)

    Returns:
        Tuple of (LLMResponse, raw_response)
    """
    if provider == "openai":
        return call_openai(messages, tools=tools, model=model or "gpt-4o",
                          temperature=temperature, base_url=base_url, api_key=api_key)
    elif provider == "qwen":
        # Qwen uses OpenAI-compatible API
        return call_openai(messages, tools=tools, model=model or "Qwen/Qwen3-32B",
                          temperature=temperature, base_url=base_url or "http://127.0.0.1:8000/v1",
                          api_key=api_key or "not-needed")
    elif provider == "anthropic":
        m = model or "claude-sonnet-4-5-20250929"
        sys_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        return call_anthropic(messages, tools=tools, model=m, system=sys_msg, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_with_tools(messages, provider="openai", model="gpt-4o", temperature=0.3,
                    max_rounds=10, verbose=True, use_tools=True, base_url=None, api_key=None):
    """
    Run LLM with tool access, handling multi-round tool calls.

    This function orchestrates a multi-turn conversation where the LLM can:
    1. Call tools to gather information
    2. Receive tool results
    3. Continue reasoning with the results
    4. Repeat until it provides a final answer

    Args:
        messages: Initial conversation messages
        provider: 'openai', 'anthropic', or 'qwen'
        model: Model name
        temperature: Sampling temperature
        max_rounds: Maximum tool-calling rounds
        verbose: Print tool calls
        use_tools: Enable tool calling (if False, single-round without tools)
        base_url: Base URL for OpenAI-compatible APIs (for Qwen)
        api_key: API key (optional)

    Returns:
        Tuple of (final_text, tool_log)
        - final_text: LLM's final response text
        - tool_log: List of tool calls with args and results
    """
    from chess_tools import TOOL_SCHEMAS_OPENAI, TOOL_SCHEMAS_ANTHROPIC, execute_tool

    if use_tools:
        tool_schemas = TOOL_SCHEMAS_OPENAI if provider in ("openai", "qwen") else TOOL_SCHEMAS_ANTHROPIC
    else:
        tool_schemas = []
        max_rounds = 1
    tool_log = []

    for _ in range(max_rounds):
        resp, raw = call_llm(messages, tools=tool_schemas, provider=provider,
                             model=model, temperature=temperature, base_url=base_url, api_key=api_key)

        if not resp.tool_calls:
            return resp.text.strip() if resp.text else "", tool_log

        if provider in ("openai", "qwen"):
            messages.append({
                "role": "assistant", "content": resp.text or None,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.name,
                                             "arguments": json.dumps(tc.arguments)}}
                               for tc in resp.tool_calls],
            })
            for tc in resp.tool_calls:
                result = execute_tool(tc.name, tc.arguments)
                tool_log.append({"tool": tc.name, "args": tc.arguments, "result": result})
                if verbose:
                    print(f"    tool: {tc.name}({tc.arguments}) -> {result[:80]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        elif provider == "anthropic":
            content_blocks = []
            if resp.text:
                content_blocks.append({"type": "text", "text": resp.text})
            for tc in resp.tool_calls:
                content_blocks.append({"type": "tool_use", "id": tc.id,
                                       "name": tc.name, "input": tc.arguments})
            messages.append({"role": "assistant", "content": content_blocks})
            tool_results = []
            for tc in resp.tool_calls:
                result = execute_tool(tc.name, tc.arguments)
                tool_log.append({"tool": tc.name, "args": tc.arguments, "result": result})
                if verbose:
                    print(f"    tool: {tc.name}({tc.arguments}) -> {result[:80]}")
                tool_results.append({"type": "tool_result", "tool_use_id": tc.id,
                                     "content": result})
            messages.append({"role": "user", "content": tool_results})

    return resp.text.strip() if resp.text else "[max tool rounds]", tool_log
