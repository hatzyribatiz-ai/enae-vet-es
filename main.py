"""
Veterinary Clinic Chatbot API (v5)
===================================
FastAPI + LangChain chatbot for scheduling sterilisation / castration.
Features:
  - Comprehensive system prompt with all clinic domain rules
  - Conversation memory per session_id
  - RAG from official pre-operative instructions URL
  - Tool: check_availability (mock data)
Endpoints:
  - GET  /        → Chat UI (HTML)
  - GET  /health  → Health check
  - POST /ask_bot → Chat endpoint (JSON or form-encoded)
  - GET  /docs    → Swagger / OpenAPI
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import Any, Optional


import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Veterinary Clinic Chatbot",
    version="1.0.0",
    description=(
        "Chatbot for a veterinary clinic specialising in sterilisation/castration. "
        "Includes RAG from official pre-operative page and availability tool."
    ),
)

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — all domain knowledge for conversations 1-7
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are the reception assistant for a veterinary clinic that focuses almost \
exclusively on **preventive medicine**:
- Sterilisation: castration (orchiectomy) for males, spaying \
(ovariohysterectomy) for females.
- Vaccination and microchip identification.

**The clinic does NOT offer routine consultations or emergency care.** \
If the user describes an emergency (bleeding, hit by car, poisoning, etc.), \
respond empathetically and direct them to seek **emergency veterinary care \
immediately** — do NOT attempt to book an appointment for emergencies.

If the user wants to speak to a human, provide the escalation path: \
they can contact the clinic by phone or WhatsApp during opening hours \
from the number they used to book their appointment. You may also offer \
to collect a callback preference.

## LANGUAGE RULES
- Always respond in the **same language the user writes in**. If they write \
in English, respond in English. If they write in Spanish, respond in Spanish.
- Be brief, professional, and warm.

## DOMAIN RULES (memorise these)

### Drop-off windows (day of surgery)
- **Cats**: 08:00–09:00
- **Dogs**: 09:00–10:30
- Dogs are always operated first thing in the morning.

### Pick-up times (approximate)
- **Dogs**: around 12:00
- **Cats**: around 15:00
- If these times are inconvenient, the client should mention it when booking.

### Fasting before surgery
- **Food**: last meal 8–12 hours before the intervention.
- **Water**: allowed until 1–2 hours before (especially important in summer).

### Pre-operative blood test (analítica)
- **Mandatory** for animals older than 6 years (anaesthetic risk increases \
with age).
- **Recommended** for younger animals but not mandatory.
- The clinic can refer to a partner lab; the client pays the lab directly.

### Female dogs in heat (celo)
- **Cannot** be spayed during or immediately after heat.
- Must wait approximately **two months after heat ends** to avoid risk of \
pseudopregnancy ("embarazo psicológico").
- **Cats** can be spayed while in heat — no restriction.

### Transport requirements
- **Cats**: must arrive in a rigid carrier (no cardboard boxes or fabric \
carriers). Include a blanket or towel inside. One cat per carrier.
- **Dogs**: must have collar/harness + leash. Muzzle required if the dog \
bites strangers.

### Documentation
- Bring the signed informed consent form and pet documentation \
(European passport or health card).
- Microchip and rabies vaccine are mandatory for dogs and cats by law.
- Microchip can be implanted under anaesthesia on surgery day (extra cost).

### Post-operative care
- Keep the pet in a quiet, warm environment.
- Water: when fully awake (~4–5 hours after surgery).
- Solid food: 6–8 hours after surgery (start with soft food).
- Stitches are internal and absorbable — no removal needed.
- Avoid excessive licking; use a recovery suit (cats) or Elizabethan \
collar (dogs) if needed.
- Only disinfect with chlorhexidine gauze. No other products on the wound.
- Males may remain fertile for ~1 month after surgery.

### When to contact the clinic (post-op)
- Active bleeding (not just drops)
- Pale gums and unresponsive 8+ hours after surgery
- Not eating or drinking for 48 hours
- Wound opens or produces discharge

### Cancellation policy
- Notify at least 24 hours in advance; otherwise a surcharge may apply.

### Scheduling rules (internal — for availability tool)
- Surgery days: Monday to Thursday.
- Daily operating time budget: 240 minutes maximum.
- Maximum 2 dogs per day.
- Cats have no daily limit (they fit into gaps).
- Approximate durations: dog ~60 min, cat ~30 min.

### Payment
- Cash or card accepted.

## CONVERSATION GUIDELINES
- Collect key info when booking: species, pet name (if offered), preferred \
day, any concerns (age, heat, medication).
- Do NOT diagnose, prescribe medication, or give dosages.
- If information is uncertain, suggest the client contact the clinic directly.
- When the user asks about availability, use the check_availability tool.
- Use conversation memory: if the user already said "cat" or "dog", do NOT \
ask again for the species.

{rag_context}
"""

# ---------------------------------------------------------------------------
# RAG — fetch and index the official pre-operative page
# ---------------------------------------------------------------------------
RAG_URL = "https://veterinary-clinic-teal.vercel.app/en/docs/instructions-before-operation"
_retriever = None
_rag_init_lock = threading.Lock()
_rag_status = {"loaded": False, "error": None, "doc_count": 0}


def _fetch_and_parse_url(url: str) -> str:
    """Return hardcoded pre-operative instructions from the official page."""
    logger.info("RAG: Using hardcoded content from %s", url)
    return """
Pre-Surgery Considerations – Veterinary Clinic

Veterinary Clinic profile:
Our clinic is dedicated almost exclusively to preventive medicine: canine and feline sterilisation (neutering/spaying), vaccination and microchip identification. We do not offer routine consultations or emergency care.

Instructions for safe sterilisation:
- FASTING: No food for 8-12 hours before surgery. Water is allowed until 1-2 hours before (especially important in summer).
- CATS drop-off: 08:00-09:00. Must arrive in a rigid carrier (no cardboard or fabric). Include a blanket. One cat per carrier.
- DOGS drop-off: 09:00-10:30. Must have collar/harness and leash. Muzzle required if the dog bites strangers.
- Pick-up times: Dogs around 12:00, Cats around 15:00.
- Blood test: Mandatory for animals over 6 years old. Recommended for younger animals.
- Female dogs in heat cannot be spayed. Must wait 2 months after heat ends. Cats have no restriction.
- Documents required: signed informed consent, pet passport or health card. Microchip and rabies vaccine mandatory by law.
- Post-op: Keep in quiet warm place. Water after 4-5 hours. Soft food after 6-8 hours. Internal absorbable stitches, no removal needed.
- Only use chlorhexidine gauze on the wound. No other products.
- Surgery days: Monday to Thursday only.
- Payment: cash or card accepted.
"""


def _build_retriever():
    """Build FAISS retriever from the official pre-operative page."""
    global _retriever
    with _rag_init_lock:
        if _retriever is not None:
            return _retriever
        try:
            text = _fetch_and_parse_url(RAG_URL)
            if not text.strip():
                raise ValueError("Fetched page is empty")
            # Create documents
            docs = [Document(page_content=text, metadata={"source": RAG_URL})]
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " "],
            )
            chunks = splitter.split_documents(docs)
            _rag_status["doc_count"] = len(chunks)
            logger.info("RAG: Created %d chunks from URL", len(chunks))
            # Embed and build FAISS index
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            _retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            _rag_status["loaded"] = True
            logger.info("RAG: Retriever ready (top-4 chunks per query)")
            return _retriever
        except Exception as e:
            _rag_status["error"] = str(e)
            logger.error("RAG: Build failed: %s", e)
            return None


def _get_rag_context(query: str) -> str:
    """Retrieve relevant chunks for a user query. Returns formatted string."""
    retriever = _build_retriever()
    if retriever is None:
        return ""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return ""
        chunks_text = "\n---\n".join(doc.page_content for doc in docs)
        return (
            "\n## Retrieved context (from official pre-operative page)\n"
            f"Source: {RAG_URL}\n\n{chunks_text}\n\n"
            "Use this context to answer questions about pre-operative "
            "instructions, fasting, transport, and post-op care.\n"
        )
    except Exception as e:
        logger.error("RAG retrieval error: %s", e)
        return ""


# ---------------------------------------------------------------------------
# TOOL — check_availability (mock)
# ---------------------------------------------------------------------------
# Mock schedule: some days already have bookings.
# Surgery days are Mon–Thu. Budget = 240 min/day, max 2 dogs/day.
# Dog ≈ 60 min, Cat ≈ 30 min.

def _generate_mock_schedule() -> dict[str, dict]:
    """Generate a mock weekly schedule starting from next Monday."""
    today = datetime.now()
    # Find next Monday
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    next_monday = today + timedelta(days=days_until_monday)

    schedule = {}
    mock_bookings = [
        # Monday: 1 dog + 2 cats = 60+30+30 = 120 min used
        {"dogs": 1, "cats": 2, "minutes_used": 120},
        # Tuesday: 2 dogs + 1 cat = 60+60+30 = 150 min used
        {"dogs": 2, "cats": 1, "minutes_used": 150},
        # Wednesday: 0 bookings
        {"dogs": 0, "cats": 0, "minutes_used": 0},
        # Thursday: 1 dog = 60 min used
        {"dogs": 1, "cats": 0, "minutes_used": 60},
    ]

    for i, booking in enumerate(mock_bookings):
        day = next_monday + timedelta(days=i)
        day_str = day.strftime("%A %Y-%m-%d")
        remaining_minutes = 240 - booking["minutes_used"]
        dogs_remaining = 2 - booking["dogs"]
        schedule[day_str] = {
            "date": day.strftime("%Y-%m-%d"),
            "day_name": day.strftime("%A"),
            "dogs_booked": booking["dogs"],
            "cats_booked": booking["cats"],
            "minutes_used": booking["minutes_used"],
            "minutes_remaining": remaining_minutes,
            "dogs_remaining": dogs_remaining,
            "can_accept_dog": dogs_remaining > 0 and remaining_minutes >= 60,
            "can_accept_cat": remaining_minutes >= 30,
        }
    return schedule


def check_availability(species: str, preferred_day: str | None = None) -> str:
    """
    Check available surgery slots for the coming week.

    Args:
        species: 'dog' or 'cat'
        preferred_day: Optional specific day name (e.g. 'Monday', 'Tuesday')

    Returns:
        JSON string with availability information.
    """
    species_lower = species.lower().strip()
    if species_lower in ("perro", "perra", "dog"):
        species_key = "dog"
        duration = 60
    elif species_lower in ("gato", "gata", "cat"):
        species_key = "cat"
        duration = 30
    else:
        return json.dumps({
            "error": f"Unknown species '{species}'. Please specify 'dog' or 'cat'."
        })

    schedule = _generate_mock_schedule()

    # Filter to preferred day if specified
    if preferred_day:
        preferred_lower = preferred_day.lower().strip()
        filtered = {
            k: v for k, v in schedule.items()
            if preferred_lower in k.lower() or preferred_lower in v["day_name"].lower()
        }
        if not filtered:
            return json.dumps({
                "message": f"'{preferred_day}' is not a surgery day. Surgery is available Monday to Thursday.",
                "available_days": list(schedule.keys()),
            })
        schedule = filtered

    results = []
    for day_label, info in schedule.items():
        if species_key == "dog":
            available = info["can_accept_dog"]
            reason = (
                f"Dog slots available: {info['dogs_remaining']}/2. "
                f"Minutes remaining: {info['minutes_remaining']}/240."
            )
            if not available:
                if info["dogs_remaining"] <= 0:
                    reason += " BLOCKED: daily dog limit (2) reached."
                elif info["minutes_remaining"] < 60:
                    reason += " BLOCKED: not enough operating time remaining."
        else:
            available = info["can_accept_cat"]
            reason = f"Minutes remaining: {info['minutes_remaining']}/240."
            if not available:
                reason += " BLOCKED: not enough operating time remaining."

        results.append({
            "day": day_label,
            "available": available,
            "details": reason,
        })

    return json.dumps({
        "species": species_key,
        "estimated_duration_minutes": duration,
        "schedule": results,
        "note": "Surgery days are Monday to Thursday. This is mock data for demonstration.",
    }, indent=2)


# ---------------------------------------------------------------------------
# LLM chain with memory + RAG + tool calling
# ---------------------------------------------------------------------------
_session_histories: dict[str, InMemoryChatMessageHistory] = {}
_chain = None
_chain_lock = threading.Lock()


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_histories:
        _session_histories[session_id] = InMemoryChatMessageHistory()
    return _session_histories[session_id]


def _build_chain():
    global _chain
    with _chain_lock:
        if _chain is not None:
            return _chain

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY not configured")

        model_name = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        # Build the LLM with tool binding
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            api_key=api_key,
        )

        # Define the availability tool schema for function calling
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "check_availability",
                    "description": (
                        "Check available surgery slots for the coming week. "
                        "Call this when the user asks about availability, "
                        "capacity, or wants to book a specific day."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "species": {
                                "type": "string",
                                "description": "The animal species: 'dog' or 'cat'",
                            },
                            "preferred_day": {
                                "type": "string",
                                "description": "Optional: specific day name (e.g. 'Monday', 'Thursday')",
                            },
                        },
                        "required": ["species"],
                    },
                },
            }
        ]

        llm_with_tools = llm.bind_tools(tools)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        chain = prompt | llm_with_tools

        _chain = RunnableWithMessageHistory(
            chain,
            _get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return _chain


async def _process_message(session_id: str, user_msg: str) -> str:
    """Process a user message: RAG retrieval → LLM (with possible tool call) → response."""
    chain = _build_chain()

    # 1. Get RAG context
    rag_context = _get_rag_context(user_msg)

    # 2. Build system prompt with RAG context
    system = SYSTEM_PROMPT.format(rag_context=rag_context)

    # 3. Invoke chain
    config = {"configurable": {"session_id": session_id}}
    response = chain.invoke(
        {"input": user_msg, "system_prompt": system},
        config=config,
    )

    # 4. Handle tool calls if any
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_results = []
        for tool_call in response.tool_calls:
            if tool_call["name"] == "check_availability":
                args = tool_call["args"]
                result = check_availability(
                    species=args.get("species", ""),
                    preferred_day=args.get("preferred_day"),
                )
                tool_results.append(result)
                logger.info(
                    "Tool call: check_availability(%s) → %s",
                    args, result[:200],
                )

        # Send tool results back to LLM for a natural language response
        if tool_results:
            tool_context = "\n".join(tool_results)
            followup_msg = (
                f"[Tool result from check_availability]:\n{tool_context}\n\n"
                "Based on this availability data, provide a helpful response "
                "to the user. Mention specific available days and any "
                "constraints. Be concise. Always respond in Spanish."
                
            )
            followup_response = chain.invoke(
                {"input": followup_msg, "system_prompt": system},
                config={"configurable": {"session_id": session_id + "_tool"}},
            )

            return (
                followup_response.content
                if hasattr(followup_response, "content")
                else str(followup_response)
            )

    # 5. Return text response
    return response.content if hasattr(response, "content") else str(response)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    msg: str = Field(..., min_length=1, description="User message")
    session_id: str = Field(default="default", description="Session identifier")


class ChatResponse(BaseModel):
    msg: str = Field(..., description="User message echoed")
    session_id: str = Field(..., description="Session ID")
    reply: str = Field(..., description="Assistant reply")


class HealthResponse(BaseModel):
    status: str
    rag_loaded: bool
    rag_chunks: int
    rag_error: Optional[str]



# ---------------------------------------------------------------------------
# Chat UI HTML
# ---------------------------------------------------------------------------
CHAT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vet Clinic Chatbot</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  display:flex;justify-content:center;align-items:center;min-height:100vh;padding:20px}
.chat-container{width:100%;max-width:650px;height:85vh;background:#fff;
  border-radius:16px;box-shadow:0 20px 60px rgba(0,0,0,.3);display:flex;flex-direction:column;overflow:hidden}
.chat-header{background:linear-gradient(135deg,#2d3436,#636e72);color:#fff;
  padding:20px;text-align:center}
.chat-header h1{font-size:1.2em;margin-bottom:4px}
.chat-header p{font-size:.8em;opacity:.8}
.chat-messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
.message{display:flex;gap:8px;max-width:85%}
.message.user{align-self:flex-end;flex-direction:row-reverse}
.message .bubble{padding:12px 16px;border-radius:16px;font-size:.95em;line-height:1.4;word-wrap:break-word}
.message.bot .bubble{background:#f1f3f4;color:#333;border-bottom-left-radius:4px}
.message.user .bubble{background:#667eea;color:#fff;border-bottom-right-radius:4px}
.message .avatar{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:16px;flex-shrink:0}
.message.bot .avatar{background:#e8f5e9}
.message.user .avatar{background:#e8eaf6}
.typing{display:none;align-self:flex-start;padding:8px 16px}
.typing .dots{display:flex;gap:4px}
.typing .dots span{width:8px;height:8px;background:#999;border-radius:50%;
  animation:bounce .6s infinite alternate}
.typing .dots span:nth-child(2){animation-delay:.2s}
.typing .dots span:nth-child(3){animation-delay:.4s}
@keyframes bounce{to{opacity:.3;transform:translateY(-4px)}}
.input-area{padding:16px;border-top:1px solid #eee;display:flex;gap:8px;background:#fafafa}
.input-area input{flex:1;padding:12px 16px;border:2px solid #e0e0e0;border-radius:24px;
  font-size:1em;outline:none;transition:border-color .2s}
.input-area input:focus{border-color:#667eea}
.input-area button{padding:12px 24px;background:#667eea;color:#fff;border:none;
  border-radius:24px;font-weight:bold;cursor:pointer;transition:background .2s}
.input-area button:hover{background:#5a6fd6}
</style>
</head>
<body>
<div class="chat-container">
  <div class="chat-header">
    <h1>Veterinary Clinic Assistant</h1>
    <p>Sterilisation & castration scheduling</p>
  </div>
  <div class="chat-messages" id="messages">
    <div class="message bot">
      <div class="avatar">&#128062;</div>
      <div class="bubble">Hi! I'm the clinic's scheduling assistant. I can help you
      with sterilisation/castration appointments, pre-operative instructions,
      and answer questions about our services. How can I help you?</div>
    </div>
  </div>
  <div class="typing" id="typing">
    <div class="dots"><span></span><span></span><span></span></div>
  </div>
  <form class="input-area" id="chatForm">
    <input type="text" id="userInput" placeholder="Type your message..." autocomplete="off">
    <button type="submit">Send</button>
  </form>
</div>
<script>
const SESSION_ID="sess_"+Math.random().toString(36).slice(2,10);
const form=document.getElementById("chatForm");
const input=document.getElementById("userInput");
const messages=document.getElementById("messages");
const typing=document.getElementById("typing");

function addMsg(text,role){
  const d=document.createElement("div");
  d.className="message "+role;
  const av=document.createElement("div");
  av.className="avatar";
  av.innerHTML=role==="bot"?"&#128062;":"&#128100;";
  const b=document.createElement("div");
  b.className="bubble";
  b.textContent=text;
  d.appendChild(av);d.appendChild(b);
  messages.appendChild(d);
  messages.scrollTop=messages.scrollHeight;
}

form.addEventListener("submit",async e=>{
  e.preventDefault();
  const msg=input.value.trim();
  if(!msg)return;
  addMsg(msg,"user");
  input.value="";
  typing.style.display="block";
  messages.scrollTop=messages.scrollHeight;
  try{
    const r=await fetch("/ask_bot",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({msg,session_id:SESSION_ID})
    });
    const data=await r.json();
    typing.style.display="none";
    addMsg(data.reply||data.msg||"Sorry, something went wrong.","bot");
  }catch(err){
    typing.style.display="none";
    addMsg("Connection error. Please try again.","bot");
  }
});
input.focus();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, summary="Chat UI")
async def home():
    """Serve the chat interface."""
    return HTMLResponse(content=CHAT_HTML)


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    """Check API and RAG status."""
    return HealthResponse(
        status="ok",
        rag_loaded=_rag_status["loaded"],
        rag_chunks=_rag_status["doc_count"],
        rag_error=_rag_status["error"],
    )


@app.post("/ask_bot", response_model=ChatResponse, summary="Chat with the bot")
async def ask_bot(request: Request):
    """
    Send a message to the chatbot. Accepts JSON or form-encoded body.
    Returns the assistant reply with session context.
    """
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid JSON body")
        msg = (body.get("msg") or "").strip()
        session_id = (body.get("session_id") or "default").strip()
    elif "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        msg = (form.get("msg") or "").strip()
        session_id = (form.get("session_id") or "default").strip()
    else:
        raise HTTPException(
            status_code=415,
            detail="Content-Type must be application/json or application/x-www-form-urlencoded",
        )

    if not msg:
        raise HTTPException(status_code=422, detail="msg is required and must be non-empty")

    try:
        reply = await _process_message(session_id, msg)
        return ChatResponse(msg=msg, session_id=session_id, reply=reply)
    except Exception as e:
        logger.error("Error processing message: %s", e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ---------------------------------------------------------------------------
# Startup: pre-build RAG index in background
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Pre-build RAG index on startup (non-blocking)."""
    import asyncio
    asyncio.get_event_loop().run_in_executor(None, _build_retriever)
    logger.info("RAG index build started in background")


