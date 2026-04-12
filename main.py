"""Veterinary clinic chatbot API: GET / (HTML) and POST /ask_bot (urlencoded)."""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any
from urllib.parse import parse_qs

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(
    title="Chatbot v4",
    version="0.1.0",
    description="Clinic chatbot: sterilisation/castration scheduling via LangChain + OpenAI.",
)

STERILIZATION_SYSTEM_PROMPT = """Eres asistente de recepción de una clínica veterinaria. \
Ayudas a los clientes a reservar citas de esterilización o castración (cirugía para \
evitar reproducción: ovariohisterectomía u orquiectomía según el caso).

Normas:
- Responde SIEMPRE en español (todas tus frases, sin excepciones), aunque el cliente \
escriba en otro idioma.
- Usa "paciente" para el animal y "cliente" para el dueño o tutora/tutor.
- No diagnosticas, no prescribes ni das dosis. Ante urgencias o dudas clínicas, indica \
que llamen a la clínica o acudan a un veterinario de inmediato.
- Los horarios de quirófano son internos; explica ventanas de ingreso y ayuno. Ayuno \
habitual: sin comida 8-12 h antes; agua hasta 1-2 h antes; el personal confirmará los \
detalles.
- Gatos: ventana de ingreso habitual 08:00-09:00. Perros: habitual 09:00-10:30. \
Formula como orientación típica salvo que el cliente tenga otras indicaciones.
- Recopila lo que necesita el equipo: especie, nombre del paciente si lo ofrecen, día \
preferido, dudas (edad, celo, medicación). Sé breve y profesional.\
"""

CONVERSATION_TEMPLATE = (
    STERILIZATION_SYSTEM_PROMPT
    + "\n\nConversación hasta ahora:\n{history}\nCliente: {input}\nAsistente:"
)

_session_chains: dict[str, ConversationChain] = {}
_session_invoke_locks: dict[str, threading.Lock] = {}
_session_locks_guard = threading.Lock()
_shared_llm: ChatOpenAI | None = None
_llm_init_lock = threading.Lock()


class AskBotResponse(BaseModel):
    """POST /ask_bot: echoes the user message and returns the assistant reply."""

    msg: str = Field(examples=["hello"])
    session_id: str = Field(examples=["s1"])
    reply: str = Field(
        default="",
        description="Assistant reply (sterilisation/castration booking).",
    )


def _get_openai_api_key() -> str:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the server",
        )
    return key


def _get_llm() -> ChatOpenAI:
    """Lazy singleton for the chat model (thread-safe)."""
    global _shared_llm
    with _llm_init_lock:
        if _shared_llm is None:
            _shared_llm = ChatOpenAI(
                model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                api_key=_get_openai_api_key(),
            )
        return _shared_llm


def _conversation_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["history", "input"],
        template=CONVERSATION_TEMPLATE,
    )


def _invoke_lock_for_session(session_id: str) -> threading.Lock:
    """Return a lock so the same session is not processed concurrently from thread pool."""
    with _session_locks_guard:
        lock = _session_invoke_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _session_invoke_locks[session_id] = lock
        return lock


def _run_conversation_sync(session_id: str, user_message: str) -> str:
    """Run chain in a worker thread; serialized per session_id for memory safety."""
    with _invoke_lock_for_session(session_id):
        if session_id not in _session_chains:
            memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=False,
                human_prefix="Cliente",
                ai_prefix="Asistente",
            )
            _session_chains[session_id] = ConversationChain(
                llm=_get_llm(),
                prompt=_conversation_prompt(),
                memory=memory,
                verbose=False,
            )
        chain = _session_chains[session_id]
        result: dict[str, Any] = chain.invoke({"input": user_message})
        text = result.get("response")
        if not isinstance(text, str):
            return str(text)
        return text


async def run_conversation(session_id: str, user_message: str) -> str:
    """Run the sync LangChain chain in a thread pool."""
    return await asyncio.to_thread(_run_conversation_sync, session_id, user_message)


@app.get(
    "/",
    summary="Home",
    response_class=Response,
    responses={200: {"content": {"text/html": {}}}},
)
async def home() -> Response:
    html = (
        "<!DOCTYPE html>"
        "<html><head><title>Clinic chatbot</title></head>"
        "<body><h1>Veterinary clinic chatbot</h1>"
        "<p>Sterilisation / castration scheduling (POST <code>/ask_bot</code>).</p>"
        "</body></html>"
    )
    return Response(content=html, media_type="text/html")


def _parse_urlencoded_body(body_bytes: bytes) -> dict[str, str]:
    try:
        body_text = body_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail="Request body is not valid UTF-8",
        ) from exc
    parsed = parse_qs(body_text, keep_blank_values=True)
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
    reply = await run_conversation(session_id, msg)
    return AskBotResponse(msg=msg, session_id=session_id, reply=reply)
