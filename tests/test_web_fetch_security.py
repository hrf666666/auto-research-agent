"""Web fetch SSRF and redirect guards."""
from __future__ import annotations

import io
import json
from unittest.mock import patch

import urllib.error

from core.tools import ToolRegistry


class _Resp:
    def __init__(self, url: str, body: str = "<html><title>OK</title><body>Hello</body>"):
        self._url = url
        self._body = body.encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body

    def geturl(self):
        return self._url


def test_blocks_direct_loopback(tmp_path):
    registry = ToolRegistry(tmp_path)

    out = json.loads(registry.execute_tool("web_fetch", {"url": "http://127.0.0.1/x", "fetch_info": "x"}))

    assert "error" in out
    assert "internal/private" in out["error"]


def test_public_to_private_redirect_blocked(tmp_path, monkeypatch):
    registry = ToolRegistry(tmp_path)
    monkeypatch.setattr(registry, "_validate_public_http_url", lambda url: (False, "Blocked: internal/private URL not allowed") if "127.0.0.1" in url else (True, ""))

    def fake_open(req, timeout=15):
        raise urllib.error.HTTPError(
            req.full_url, 302, "Found", {"Location": "http://127.0.0.1/private"}, io.BytesIO()
        )

    with patch("urllib.request.OpenerDirector.open", side_effect=fake_open):
        out = json.loads(registry.execute_tool("web_fetch", {"url": "https://example.com/start", "fetch_info": "x"}))

    assert "error" in out
    assert "internal/private" in out["error"]


def test_relative_public_redirect_allowed(tmp_path, monkeypatch):
    registry = ToolRegistry(tmp_path)
    monkeypatch.setattr(registry, "_validate_public_http_url", lambda url: (True, ""))
    calls = {"n": 0}

    def fake_open(req, timeout=15):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 302, "Found", {"Location": "/next"}, io.BytesIO())
        return _Resp(req.full_url, "<html><title>Next</title><body>Body</body>")

    with patch("urllib.request.OpenerDirector.open", side_effect=fake_open):
        out = json.loads(registry.execute_tool("web_fetch", {"url": "https://example.com/start", "fetch_info": "x"}))

    assert out["title"] == "Next"
    assert out["url"] == "https://example.com/next"
