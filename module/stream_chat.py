import os
import random
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI


def usage_to_dict(usage_obj: Any) -> Dict[str, Any]:
    """Convert provider usage objects to plain dicts without dropping nested details."""
    if usage_obj is None:
        return {}
    if isinstance(usage_obj, dict):
        return dict(usage_obj)
    if hasattr(usage_obj, "model_dump"):
        dumped = usage_obj.model_dump(exclude_none=True)
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(usage_obj, "to_dict"):
        dumped = usage_obj.to_dict()
        return dumped if isinstance(dumped, dict) else {}

    usage: Dict[str, Any] = {}
    for field in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "input_tokens_details",
        "output_tokens_details",
    ):
        value = getattr(usage_obj, field, None)
        if value is not None:
            usage[field] = usage_to_dict(value) if not isinstance(value, (int, float, str, bool)) else value
    return usage


def get_cached_prompt_tokens(usage: Optional[Dict[str, Any]]) -> int:
    """Extract prompt/input cache-hit tokens from common OpenAI-compatible usage schemas."""
    if not usage:
        return 0

    candidate_paths = (
        ("prompt_tokens_details", "cached_tokens"),
        ("input_tokens_details", "cached_tokens"),
        ("input_token_details", "cache_read"),
        ("input_tokens_details", "cache_read"),
        ("prompt_tokens_details", "cache_read"),
    )
    total = 0
    for parent_key, child_key in candidate_paths:
        parent = usage.get(parent_key)
        if isinstance(parent, dict):
            value = parent.get(child_key)
            if isinstance(value, int):
                total += value

    for key in (
        "cached_tokens",
        "cached_prompt_tokens",
        "prompt_cache_hit_tokens",
        "cache_read_input_tokens",
        "input_cache_hit_tokens",
    ):
        value = usage.get(key)
        if isinstance(value, int):
            total += value
    return total


def _uses_openai_gpt5_chat_constraints(base_url: str, model: str) -> bool:
    """Return True for official OpenAI GPT-5 chat models with stricter params."""
    normalized_model = model.rsplit("/", 1)[-1]
    return "api.openai.com" in base_url.lower() and normalized_model.startswith("gpt-5")


def _normalize_model_for_base_url(base_url: str, model: str) -> str:
    """Convert provider-prefixed model names when calling official OpenAI directly."""
    if "api.openai.com" in base_url.lower() and model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def stream_chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    service_tier: Optional[str] = None,
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
    retry_max_delay: float = 10.0,
    retry_jitter: float = 0.3,
) -> tuple[str, str, str, Optional[Dict[str, Any]]]:
    """
    Stream a chat completion with automatic retry.

    max_retries is the number of retries after the first attempt.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if service_tier is None:
        service_tier = os.environ.get("OPENAI_SERVICE_TIER") or os.environ.get("SERVICE_TIER")

    base_url = str(getattr(client, "base_url", "") or "")
    enable_thinking = "dashscope.aliyuncs.com" in base_url
    openai_gpt5_chat = _uses_openai_gpt5_chat_constraints(base_url, model)
    request_model = _normalize_model_for_base_url(base_url, model)

    def _is_retryable_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            if status_code in {408, 409, 429} or status_code >= 500:
                return True

        err_text = f"{type(exc).__name__}: {exc}".lower()
        retry_markers = (
            "timeout",
            "timed out",
            "rate limit",
            "connection",
            "network",
            "temporarily unavailable",
            "service unavailable",
            "try again",
            "overloaded",
            "stream closed",
        )
        return any(marker in err_text for marker in retry_markers)

    def _run_once() -> tuple[str, str, str, Optional[Dict[str, Any]]]:
        request_kwargs: Dict[str, Any] = {
            "model": request_model,
            "messages": messages,
            # stop=["\n<tool_response>", "<tool_response>"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "timeout": 60000,
        }
        if openai_gpt5_chat:
            request_kwargs["max_completion_tokens"] = 10000
        else:
            request_kwargs.update(
                {
                    "top_p": 0.95,
                    "temperature": 0.6,
                    "max_tokens": 10000,
                    "presence_penalty": 1.1,
                }
            )
        if enable_thinking:
            request_kwargs["extra_body"] = {"enable_thinking": True}
        if service_tier:
            request_kwargs["service_tier"] = service_tier

        stream = client.chat.completions.create(**request_kwargs)

        content_parts: List[str] = []
        reasoning_content = ""
        aggregated_calls: Dict[str, Dict[str, Any]] = {}
        usage: Optional[Dict[str, Any]] = None
        served_tier_seen: Optional[str] = None

        for chunk in stream:
            served_tier = getattr(chunk, "service_tier", None)
            if served_tier:
                served_tier_seen = served_tier
                usage = usage or {}
                usage["service_tier"] = served_tier_seen

            usage_obj = getattr(chunk, "usage", None)
            if usage_obj:
                usage = usage_to_dict(usage_obj)
                if served_tier_seen:
                    usage["service_tier"] = served_tier_seen

            for choice in getattr(chunk, "choices", []) or []:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                delta_reason = getattr(delta, "reasoning_content", None)
                if delta_reason:
                    reasoning_content += delta_reason

                delta_content = getattr(delta, "content", None)
                if delta_content:
                    if isinstance(delta_content, list):
                        content_parts.extend([str(c) for c in delta_content])
                    else:
                        content_parts.append(str(delta_content))

                for tc in getattr(delta, "tool_calls", []) or []:
                    tc_id = getattr(tc, "id", None) or f"call_{len(aggregated_calls)}"
                    fn = getattr(tc, "function", None) or {}
                    fn_name = getattr(fn, "name", "") if hasattr(fn, "name") else fn.get("name", "")
                    fn_args = getattr(fn, "arguments", None) if hasattr(fn, "arguments") else fn.get("arguments")

                    entry = aggregated_calls.get(
                        tc_id,
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": fn_name, "arguments": ""},
                        },
                    )
                    if fn_name:
                        entry["function"]["name"] = fn_name
                    if fn_args is not None:
                        if isinstance(fn_args, str):
                            entry["function"]["arguments"] = str(entry["function"].get("arguments", "")) + fn_args
                        else:
                            entry["function"]["arguments"] = fn_args

                    aggregated_calls[tc_id] = entry

        final_content = "".join(content_parts).strip()
        full_content = reasoning_content + final_content
        return full_content, final_content, reasoning_content, usage

    last_error: Optional[Exception] = None
    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            return _run_once()
        except Exception as exc:
            last_error = exc
            should_retry = attempt < total_attempts and _is_retryable_error(exc)
            if not should_retry:
                raise

            delay = min(retry_base_delay * (2 ** (attempt - 1)), retry_max_delay)
            if retry_jitter > 0:
                delay += random.uniform(0, retry_jitter)
            time.sleep(delay)

    raise RuntimeError(f"stream_chat failed after {total_attempts} attempts") from last_error
