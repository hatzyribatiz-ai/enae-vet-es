"""Chatbot v4 placeholder API (VETES-16): GET / (HTML) and POST /ask_bot (urlencoded)."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

app = FastAPI(
    title="Chatbot v4",
    version="0.1.0",
    description="Placeholder API for clinic chatbot: HTML home + form-based ask endpoint.",
)


class AskBotResponse(BaseModel):
    """Stub response for POST /ask_bot until LangChain is added."""

    msg: str = Field(examples=["hello"])
    session_id: str = Field(examples=["s1"])
    placeholder: bool = True


@app.get(
    "/",
    summary="Home",
    response_class=Response,
    responses={200: {"content": {"text/html": {}}}},
)
async def home() -> Response:
    html = (
        "<!DOCTYPE html>"
        "<html><head><title>Chatbot v4</title></head>"
        "<body><h1>Chatbot v4 Placeholder</h1>"
        "<p>This route is reserved for a future visual chat UI.</p>"
        "</body></html>"
    )
    return Response(content=html, media_type="text/html")


def _parse_urlencoded_body(body_bytes: bytes) -> dict[str, str]:
    parsed = parse_qs(body_bytes.decode("utf-8"), keep_blank_values=True)
    flat: dict[str, str] = {}
    for key, values in parsed.items():
        if values:
            flat[key] = values[-1]
    return flat


def _validate_ask_bot_fields(msg: str | None, session_id: str | None) -> tuple[str, str]:
    if msg is None or session_id is None:
        raise HTTPException(
            status_code=422,
            detail="msg and session_id are required fields",
        )
    msg_clean = msg.strip()
    session_clean = session_id.strip()
    if not msg_clean or not session_clean:
        raise HTTPException(
            status_code=422,
            detail="msg and session_id must be non-empty strings",
        )
    return msg_clean, session_clean


@app.post(
    "/ask_bot",
    summary="Ask Bot",
    response_model=AskBotResponse,
)
async def ask_bot(request: Request) -> AskBotResponse:
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/x-www-form-urlencoded" not in content_type:
        raise HTTPException(
            status_code=415,
            detail="Content-Type must be application/x-www-form-urlencoded",
        )
    body_bytes = await request.body()
    if not body_bytes.strip():
        raise HTTPException(status_code=422, detail="Request body is empty")
    fields = _parse_urlencoded_body(body_bytes)
    msg, session_id = _validate_ask_bot_fields(
        fields.get("msg"),
        fields.get("session_id"),
    )
    return AskBotResponse(msg=msg, session_id=session_id, placeholder=True)
