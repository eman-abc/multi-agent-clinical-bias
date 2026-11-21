"""
p1v2/llm.py -- one chat-completions client interface, the real OpenRouter implementation, and a scriptable fake for tests.

CallResult.error is set (and text empty) for a content-filter refusal: that is data, not a crash, and is never retried.
A guardrail / routing error (HTTP 404 'blocked by guardrail', 401/403) raises GuardrailError at once: retrying cannot fix it and 1,500 identical 404s are not data.
Timeouts, 429 and 5xx are retried with exponential backoff, and every attempt that failed is reported in `retries`.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol


class GuardrailError(RuntimeError):
    pass


@dataclass
class CallResult:
    text: str = ""
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    provider: str | None = None
    model: str | None = None
    latency_s: float = 0.0
    error: str | None = None
    retries: list = field(default_factory=list)


class Client(Protocol):
    def chat(self, messages: list[dict], *, model: str, temperature: float, max_tokens: int, extra_body: dict | None = None) -> CallResult: ...


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, base_url: str = "https://openrouter.ai/api/v1", timeout: float = 60.0, max_retries: int = 8):
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except Exception:  # noqa: BLE001
            pass
        from openai import OpenAI                      # imported lazily: the tests and the fake client do not need it
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self._c = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
        self.max_retries = max_retries

    def chat(self, messages, *, model, temperature, max_tokens, extra_body=None) -> CallResult:
        retries = []
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                r = self._c.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, extra_body=extra_body or None)
                msg = r.choices[0].message
                extra = getattr(r, "model_extra", None) or {}
                usage = r.usage.model_dump() if getattr(r, "usage", None) is not None else {}
                return CallResult(text=(msg.content or ""), reasoning=getattr(msg, "reasoning", None) or (getattr(msg, "model_extra", None) or {}).get("reasoning"), finish_reason=r.choices[0].finish_reason,
                                  usage=usage, provider=extra.get("provider"), model=getattr(r, "model", None), latency_s=time.time() - t0, retries=retries)
            except Exception as e:  # noqa: BLE001
                m = str(e)
                low = m.lower()
                if "content_filter" in low or "responsibleaipolicyviolation" in low:
                    return CallResult(error="content_filter: " + m[:300], latency_s=time.time() - t0, retries=retries)
                if "guardrail" in low or "error code: 404" in low or "error code: 401" in low or "error code: 403" in low:
                    raise GuardrailError(m[:400]) from e
                retries.append(f"attempt {attempt + 1}: {type(e).__name__}: {m[:160]}")
                if attempt == self.max_retries:
                    return CallResult(error=f"failed after {attempt + 1} attempts: {m[:300]}", latency_s=time.time() - t0, retries=retries)
                wait = min((6 if ("429" in m or "rate" in low or "overloaded" in low) else 1) * 2 ** attempt, 90)
                time.sleep(wait + random.uniform(0, 1.5))              # rate limits back off longer; jitter stops the workers retrying in step
        return CallResult(error="unreachable")


class FakeClient:
    """script(messages, call_index) -> str text, or (text, error), or a CallResult. Thread-safe call counter."""

    def __init__(self, script: Callable):
        import threading
        self.script, self.n, self._lock, self.requests = script, 0, threading.Lock(), []

    def chat(self, messages, *, model, temperature, max_tokens, extra_body=None) -> CallResult:
        with self._lock:
            self.n += 1
            i = self.n
            self.requests.append([dict(m) for m in messages])
        out = self.script(messages, i)
        if isinstance(out, CallResult):
            return out
        if isinstance(out, tuple):
            return CallResult(text=out[0], error=out[1], usage={"prompt_tokens": 10, "completion_tokens": len(out[0].split())}, provider="fake", model=model)
        return CallResult(text=out, usage={"prompt_tokens": 10, "completion_tokens": len(out.split())}, provider="fake", model=model)
