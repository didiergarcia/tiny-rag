"""OpenAICompatibleLLM against a local mock HTTP server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tiny_rag import OpenAICompatibleLLM


class MockServer:
    def __init__(self, status=200, body=None):
        self.status = status
        default = {"choices": [{"message": {"content": " Eagles "}}]}
        self.body = body if body is not None else default
        self.requests = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                server.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "json": json.loads(self.rfile.read(length)),
                    }
                )
                payload = json.dumps(server.body).encode()
                self.send_response(server.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def make_server():
    servers = []

    def make(**kwargs):
        servers.append(MockServer(**kwargs))
        return servers[-1]

    yield make
    for s in servers:
        s.close()


MESSAGES = [{"role": "user", "content": "Who won?"}]


def test_sends_chat_completion_and_returns_content(make_server):
    server = make_server()
    llm = OpenAICompatibleLLM(server.url + "/", "tinyllama", api_key="secret")

    assert llm.generate(MESSAGES, max_new_tokens=12) == "Eagles"

    request = server.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer secret"
    assert request["json"] == {
        "model": "tinyllama",
        "messages": MESSAGES,
        "max_tokens": 12,
        "temperature": 0,
    }


def test_no_auth_header_without_key(make_server):
    server = make_server()
    OpenAICompatibleLLM(server.url, "m").generate(MESSAGES)
    assert "Authorization" not in server.requests[0]["headers"]


def test_http_error_raises_with_status(make_server):
    server = make_server(status=500, body={"error": "boom"})
    with pytest.raises(RuntimeError, match="HTTP 500.*boom"):
        OpenAICompatibleLLM(server.url, "m").generate(MESSAGES)


def test_malformed_response_raises(make_server):
    server = make_server(body={"unexpected": True})
    with pytest.raises(RuntimeError, match="unexpected response"):
        OpenAICompatibleLLM(server.url, "m").generate(MESSAGES)


def test_unreachable_server_raises():
    llm = OpenAICompatibleLLM("http://127.0.0.1:9/v1", "m", timeout=2)
    with pytest.raises(RuntimeError, match="could not reach"):
        llm.generate(MESSAGES)


def test_rejects_non_http_base_url():
    with pytest.raises(ValueError):
        OpenAICompatibleLLM("localhost:8000/v1", "m")
