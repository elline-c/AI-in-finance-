"""Minimal client for a locally hosted Ollama server.

* **Local only.** The base URL must be a loopback address, so statement data cannot leave
  the machine by configuration accident. ``REVIEWER_LLM_ALLOW_REMOTE=1`` overrides this on
  purpose.
* **No proxies.** Requests ignore ``http_proxy``-style variables.
* **Standard library only.** No new dependency.
* **Deterministic as far as the server allows**: temperature 0 and a fixed seed.
* **Structured output.** The JSON schema is passed in ``format`` so the server constrains
  generation; the caller still validates the result.
* **Errors carry only the exception type**, never the prompt or the response text.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Mapping, Optional

DEFAULT_URL = "http://127.0.0.1:11434"
# 9B class, 6.6 GB; fits a 16 GB machine. The GGUF build is chosen on purpose: the default
# "qwen3.5:9b" tag runs on Ollama's MLX engine, which (in the Homebrew build tested) cannot
# do JSON-schema output, and this layer needs it.
DEFAULT_MODEL = "qwen3.5:9b-q4_K_M"
DEFAULT_TIMEOUT_S = 120.0
MAX_RESPONSE_BYTES = 2_000_000
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}

URL_ENV = "REVIEWER_LLM_URL"
MODEL_ENV = "REVIEWER_LLM_MODEL"
TIMEOUT_ENV = "REVIEWER_LLM_TIMEOUT_S"
REMOTE_ENV = "REVIEWER_LLM_ALLOW_REMOTE"


class LLMUnavailable(Exception):
    """The model server could not be reached or did not answer in time."""


class LLMError(Exception):
    """The server answered, but not with a usable result."""


@dataclass(frozen=True)
class LLMResult:
    content: str
    seconds: float
    model: str


def _truthy(v: Optional[str]) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def check_local(url: str, allow_remote: bool = False) -> str:
    """Return ``url`` without a trailing slash, or raise ValueError if it is not loopback."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("the model URL must be http(s)://host[:port]")
    if parsed.hostname.lower() not in _LOOPBACK and not allow_remote:
        raise ValueError(
            f"{parsed.hostname!r} is not a loopback address; statement data must stay on this "
            f"machine. Set {REMOTE_ENV}=1 to allow a remote server on purpose.")
    return url.rstrip("/")


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 timeout: Optional[float] = None, allow_remote: Optional[bool] = None,
                 opener=None, env: Optional[Mapping[str, str]] = None) -> None:
        env = os.environ if env is None else env
        self.allow_remote = _truthy(env.get(REMOTE_ENV)) if allow_remote is None else allow_remote
        self.base_url = check_local(base_url or env.get(URL_ENV) or DEFAULT_URL, self.allow_remote)
        self.model = model or env.get(MODEL_ENV) or DEFAULT_MODEL
        try:
            self.timeout = float(timeout if timeout is not None
                                 else env.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_S))
        except ValueError:
            self.timeout = DEFAULT_TIMEOUT_S
        self._opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # ------------------------------------------------------------------ #
    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + "/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST")
        with self._opener.open(req, timeout=self.timeout) as resp:      # noqa: S310 - checked above
            body = resp.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise LLMError("response too large")
        try:
            return json.loads(body)
        except ValueError as exc:
            raise LLMError("response is not JSON") from exc

    def chat_json(self, messages: list[dict], schema: dict, *, seed: int = 7,
                  max_tokens: int = 1200, num_ctx: int = 8192) -> LLMResult:
        """One non-streaming chat call whose answer must follow ``schema``."""
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            "format": schema, "think": False, "keep_alive": "10m",
            "options": {"temperature": 0, "seed": seed, "num_predict": max_tokens,
                        "num_ctx": num_ctx},
        }
        started = time.monotonic()
        try:
            try:
                data = self._post(payload)
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and "think" in exc.read(2000).decode("utf-8", "ignore").lower():
                    payload.pop("think")          # a model without a thinking switch
                    data = self._post(payload)
                else:
                    raise
        except urllib.error.HTTPError as exc:
            raise LLMError(f"HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise LLMUnavailable(type(exc).__name__) from None
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("empty answer")
        if data.get("done_reason") == "length":
            raise LLMError("answer was cut off")
        return LLMResult(content=content, seconds=time.monotonic() - started,
                         model=str(data.get("model") or self.model))
