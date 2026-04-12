"""Tests for main.py: FastAPI + LangChain chatbot."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_get_home_returns_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"].lower()
    html_body = resp.text
    assert html_body.strip()
    assert "chatbot" in html_body.lower()


@patch("main.run_conversation", new_callable=AsyncMock)
def test_post_ask_bot_urlencoded_ok(mock_run: AsyncMock, client: TestClient) -> None:
    mock_run.return_value = "Thank you for your question about scheduling."
    resp = client.post(
        "/ask_bot",
        content=b"msg=hello&session_id=s1",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["msg"] == "hello"
    assert data["session_id"] == "s1"
    assert data["reply"] == "Thank you for your question about scheduling."
    mock_run.assert_awaited_once_with("s1", "hello")


def test_post_ask_bot_missing_msg(client: TestClient) -> None:
    resp = client.post(
        "/ask_bot",
        content=b"session_id=s1",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 422


def test_post_ask_bot_missing_session_id(client: TestClient) -> None:
    resp = client.post(
        "/ask_bot",
        content=b"msg=hello",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 422


def test_post_ask_bot_whitespace_msg(client: TestClient) -> None:
    resp = client.post(
        "/ask_bot",
        content=b"msg=+++&session_id=s1",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 422


def test_post_ask_bot_empty_body(client: TestClient) -> None:
    resp = client.post(
        "/ask_bot",
        content=b"",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 422


def test_post_ask_bot_invalid_utf8_body(client: TestClient) -> None:
    resp = client.post(
        "/ask_bot",
        content=b"msg=\xff&session_id=s1",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 422


def test_post_ask_bot_json_unsupported(client: TestClient) -> None:
    resp = client.post(
        "/ask_bot",
        json={"msg": "hello", "session_id": "s1"},
    )
    assert resp.status_code == 415


def test_openapi_json_contains_paths(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    openapi = resp.json()
    paths = openapi.get("paths", {})
    assert "/" in paths
    assert "/ask_bot" in paths
    get_home = paths["/"].get("get", {})
    assert get_home.get("summary") == "Home"
    post_ask = paths["/ask_bot"].get("post", {})
    assert post_ask.get("summary") == "Ask Bot"
