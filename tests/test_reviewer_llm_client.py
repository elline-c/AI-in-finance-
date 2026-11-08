"""Step 5: the local model client. Uses a throw-away server on 127.0.0.1, never the internet."""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fraud_detect.reviewer import llm_client
from fraud_detect.reviewer.llm_client import LLMError, LLMUnavailable, OllamaClient, check_local


class _Server:
    def __init__(self):
        self.requests: list[dict] = []
        self.script: list = []          # each item: (status, body dict | str, delay seconds)
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):          # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append({"path": self.path, "json": json.loads(body)})
                status, payload, delay = outer.script.pop(0) if outer.script else (200, outer.ok(), 0)
                time.sleep(delay)
                data = payload if isinstance(payload, str) else json.dumps(payload)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data.encode())

            def log_message(self, *a):  # silence
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @staticmethod
    def ok(content='{"ok": true}', **extra):
        return {"model": "qwen3.5:9b", "message": {"role": "assistant", "content": content},
                "done": True, "done_reason": "stop", **extra}

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def server():
    s = _Server()
    yield s
    s.close()


MSGS = [{"role": "user", "content": "hello"}]
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://localhost:11434/", "http://[::1]:11434"])
def test_loopback_urls_are_accepted(url):
    assert check_local(url) == url.rstrip("/")


@pytest.mark.parametrize("url", ["http://192.168.1.20:11434", "https://api.example.com", "http://10.0.0.5"])
def test_remote_urls_are_refused_unless_allowed_on_purpose(url):
    with pytest.raises(ValueError, match="loopback"):
        check_local(url)
    assert check_local(url, allow_remote=True)


def test_bad_scheme_is_refused():
    with pytest.raises(ValueError):
        check_local("ftp://127.0.0.1")


def test_environment_can_allow_a_remote_server_on_purpose():
    c = OllamaClient(env={llm_client.URL_ENV: "http://10.1.1.1:11434", llm_client.REMOTE_ENV: "1"})
    assert c.base_url == "http://10.1.1.1:11434"
    with pytest.raises(ValueError):
        OllamaClient(env={llm_client.URL_ENV: "http://10.1.1.1:11434"})


def test_defaults_come_from_the_environment():
    c = OllamaClient(env={llm_client.MODEL_ENV: "qwen3.5:4b", llm_client.TIMEOUT_ENV: "7"})
    assert c.model == "qwen3.5:4b" and c.timeout == 7.0 and c.base_url == llm_client.DEFAULT_URL
    assert OllamaClient(env={}).model == llm_client.DEFAULT_MODEL == "qwen3.5:9b-q4_K_M"
    assert OllamaClient(env={llm_client.TIMEOUT_ENV: "junk"}).timeout == llm_client.DEFAULT_TIMEOUT_S


def test_request_is_constrained_deterministic_and_non_streaming(server):
    res = OllamaClient(base_url=server.url).chat_json(MSGS, SCHEMA, seed=11, max_tokens=50)
    sent = server.requests[0]
    assert sent["path"] == "/api/chat"
    body = sent["json"]
    assert body["format"] == SCHEMA and body["stream"] is False and body["think"] is False
    assert body["options"]["temperature"] == 0 and body["options"]["seed"] == 11
    assert body["options"]["num_predict"] == 50 and body["messages"] == MSGS
    assert res.content == '{"ok": true}' and res.model == "qwen3.5:9b" and res.seconds >= 0


def test_a_model_without_a_thinking_switch_is_retried_without_it(server):
    server.script = [(400, {"error": '"qwen" does not support think'}, 0)]
    res = OllamaClient(base_url=server.url).chat_json(MSGS, SCHEMA)
    assert res.content
    assert "think" in server.requests[0]["json"] and "think" not in server.requests[1]["json"]


def test_other_http_errors_become_llm_error_without_the_body(server):
    server.script = [(500, {"error": "secret prompt text leaked here"}, 0)]
    with pytest.raises(LLMError) as exc:
        OllamaClient(base_url=server.url).chat_json(MSGS, SCHEMA)
    assert "500" in str(exc.value) and "secret" not in str(exc.value)


@pytest.mark.parametrize("payload, message", [
    (_Server.ok(content=""), "empty"),
    (_Server.ok(done_reason="length") | {"done_reason": "length"}, "cut off"),
    ("not json at all", "not JSON"),
])
def test_unusable_answers_raise_llm_error(server, payload, message):
    server.script = [(200, payload, 0)]
    with pytest.raises(LLMError, match=message):
        OllamaClient(base_url=server.url).chat_json(MSGS, SCHEMA)


def test_an_unreachable_server_is_reported_as_unavailable_with_the_type_only():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(LLMUnavailable) as exc:
        OllamaClient(base_url=f"http://127.0.0.1:{port}", timeout=2).chat_json(MSGS, SCHEMA)
    assert "hello" not in str(exc.value)


def test_a_slow_server_times_out_as_unavailable(server):
    server.script = [(200, _Server.ok(), 1.5)]
    with pytest.raises(LLMUnavailable):
        OllamaClient(base_url=server.url, timeout=0.3).chat_json(MSGS, SCHEMA)


def test_proxy_settings_are_ignored(server, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    assert OllamaClient(base_url=server.url).chat_json(MSGS, SCHEMA).content
