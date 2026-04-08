"""LLM abstraction — supports OpenAI, Anthropic, and compatible APIs."""

from __future__ import annotations

import os
import json
from typing import Optional


def call_llm(
    prompt: str,
    provider: str = "openrouter",
    model: str = "openai/gpt-4.1-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    system: Optional[str] = None,
) -> str:
    """Call an LLM and return the text response."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if provider == "openrouter":
        return _call_openrouter(prompt, model, api_key, temperature, max_tokens, system)
    elif provider == "openai":
        return _call_openai(prompt, model, api_key, temperature, max_tokens, system)
    elif provider == "anthropic":
        return _call_anthropic(prompt, model, api_key, temperature, max_tokens, system)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _call_openrouter(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    system: Optional[str],
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _call_openai(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    system: Optional[str],
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _call_anthropic(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    system: Optional[str],
) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text
