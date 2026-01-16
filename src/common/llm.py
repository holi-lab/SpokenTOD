import asyncio
import os
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger
from openai import AsyncOpenAI, OpenAI


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    cost: float = 0.0
    upstream_inference_cost: float = 0.0

    def add(self, usage: dict[str, Any]) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)
        self.cost += float(usage.get("cost") or 0.0)

        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        cost_details = usage.get("cost_details") or {}

        self.cached_tokens += int(prompt_details.get("cached_tokens") or 0)
        self.audio_tokens += int(prompt_details.get("audio_tokens") or 0)
        self.reasoning_tokens += int(completion_details.get("reasoning_tokens") or 0)
        self.upstream_inference_cost += float(cost_details.get("upstream_inference_cost") or 0.0)


_USAGE_TOTALS = UsageTotals()


def _should_include_usage(include_usage: bool | None) -> bool:
    if include_usage is not None:
        return include_usage
    return os.environ.get("LLM_USAGE") == "1"


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage

    def _details(obj: Any) -> dict[str, Any] | None:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        return obj.__dict__

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "cost": getattr(usage, "cost", None),
        "prompt_tokens_details": _details(getattr(usage, "prompt_tokens_details", None)),
        "completion_tokens_details": _details(getattr(usage, "completion_tokens_details", None)),
        "cost_details": _details(getattr(usage, "cost_details", None)),
    }


def _record_usage(usage: Any, model: str = "", provider: str = "") -> None:
    usage_dict = _usage_to_dict(usage)
    if not usage_dict:
        return
    _USAGE_TOTALS.add(usage_dict)
    prompt_tokens = usage_dict.get("prompt_tokens", 0)
    completion_tokens = usage_dict.get("completion_tokens", 0)
    total_tokens = usage_dict.get("total_tokens", 0)
    cost = usage_dict.get("cost")

    if cost is None and provider == "openai":
        cost_str = "N/A (OpenAI does not provide cost)"
    else:
        cost_str = str(cost)

    logger.debug(
        f"LLM usage: prompt={prompt_tokens} completion={completion_tokens} total={total_tokens} cost={cost_str}",
    )


def get_usage_totals() -> dict[str, Any]:
    return {
        "prompt_tokens": _USAGE_TOTALS.prompt_tokens,
        "completion_tokens": _USAGE_TOTALS.completion_tokens,
        "total_tokens": _USAGE_TOTALS.total_tokens,
        "cached_tokens": _USAGE_TOTALS.cached_tokens,
        "reasoning_tokens": _USAGE_TOTALS.reasoning_tokens,
        "audio_tokens": _USAGE_TOTALS.audio_tokens,
        "cost": _USAGE_TOTALS.cost,
        "upstream_inference_cost": _USAGE_TOTALS.upstream_inference_cost,
    }


def _get_llm_client(
    provider: Literal["openrouter", "vllm", "openai"] = "openrouter",
    base_url: str = "http://localhost:8000/v1",
):
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
        )

    elif provider == "vllm":
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        return OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openrouter' or 'vllm'")


def _get_async_llm_client(
    provider: Literal["openrouter", "vllm"] = "openrouter",
    base_url: str = "http://localhost:8000/v1",
):
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        return AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    elif provider == "vllm":
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openrouter' or 'vllm'")


def completion(
    messages: list[dict[str, str]],
    model: str,
    provider: str = "openrouter",
    base_url: str = "http://localhost:8000/v1",
    include_usage: bool | None = None,
    **params,
) -> Any:
    """
    Unified LLM completion function (Synchronous).
    """
    client = _get_llm_client(provider=provider, base_url=base_url)
    if _should_include_usage(include_usage):
        extra_body = params.pop("extra_body", {}) or {}
        extra_body = {**extra_body, "usage": {"include": True}}
        params["extra_body"] = extra_body

    response = client.chat.completions.create(model=model, messages=messages, **params)
    _record_usage(getattr(response, "usage", None), model=model, provider=provider)
    return response.choices[0].message.content


async def async_completion(
    messages: list[dict[str, str]],
    model: str,
    provider: str = "openrouter",
    base_url: str = "http://localhost:8000/v1",
    include_usage: bool | None = None,
    **params,
) -> Any:
    """
    Unified LLM completion function (Asynchronous).
    """
    client = _get_async_llm_client(provider=provider, base_url=base_url)
    if _should_include_usage(include_usage):
        extra_body = params.pop("extra_body", {}) or {}
        extra_body = {**extra_body, "usage": {"include": True}}
        params["extra_body"] = extra_body

    response = await client.chat.completions.create(model=model, messages=messages, **params)
    _record_usage(getattr(response, "usage", None), model=model, provider=provider)
    return response.choices[0].message.content


async def batch_completion(
    requests: list[dict[str, Any]],
    provider: str = "openrouter",
    base_url: str = "http://localhost:8000/v1",
    include_usage: bool | None = None,
) -> list[Any]:
    """
    Execute multiple LLM completions in parallel.

    Args:
        requests: List of dicts, each containing 'messages', 'model', and other params.
        provider: Provider to use
        base_url: Base URL for provider
    """
    tasks = []
    for req in requests:
        params = {k: v for k, v in req.items() if k not in ["messages", "model"]}
        tasks.append(
            async_completion(
                messages=req["messages"],
                model=req["model"],
                provider=provider,
                base_url=base_url,
                include_usage=include_usage,
                **params,
            )
        )
    return await asyncio.gather(*tasks)
