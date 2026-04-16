# enae-vet-es — Veterinary Clinic Chatbot

Chatbot MVP for a veterinary clinic specialising in **sterilisation and castration** scheduling. Built as a case study for ENAE Business School — *Data Science e IA para la Toma de Decisiones*.

**Team**: Hachi  
**Repo**: `enae-vet-es` (fork of `kuuli/veterinary-clinic-chatbot`)  
**Jira**: https://hatzyribatiz.atlassian.net/jira/software/projects/ENAE/boards/2


---

## What's implemented

| Feature | Ticket | Status |
|---------|--------|--------|
| FastAPI with 2 endpoints + Swagger | VET-7 | Done |
| Chat UI (HTML) | VET-8 | Done |
| LangChain + system prompt | VET-9 | Done |
| Conversation memory per `session_id` | VET-10 | Done |
| RAG from official URL | VET-11 | Done |
| Tool: check_availability (mock) | VET-12 | Done |
| Intents catalog (20 intents) | VET-5 | Done |
| Vercel deploy | VET-3 | Done |
| Jira board | VET-4 | Done |

---

## Quick start (local)

```bash
# 1. Clone the repo
git clone https://github.com/hatzyribatiz-ai/enae-vet-es.git
cd enae-vet-es

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key:
#   OPENAI_API_KEY=sk-proj-...

# 5. Start the server
uvicorn main:app --reload
```

Then open:

- **Chat UI**: http://127.0.0.1:8000/
- **Swagger docs**: http://127.0.0.1:8000/docs
- **Health check**: http://127.0.0.1:8000/health

### Example API call

```bash
# JSON format
curl -X POST http://127.0.0.1:8000/ask_bot \
  -H "Content-Type: application/json" \
  -d '{"msg": "Hi, what can you help me with?", "session_id": "test1"}'

# Form-encoded format
curl -X POST http://127.0.0.1:8000/ask_bot \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "msg=Hello&session_id=test1"
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key (never commit this) |
| `OPENAI_CHAT_MODEL` | No | Model name (default: `gpt-4o-mini`) |

A `.env.example` file is provided as a template. Copy it to `.env` and fill in your key.

---

## Architecture

```
main.py              ← FastAPI app (endpoints, LLM chain, RAG, tool)
api/index.py         ← Vercel serverless entry point
requirements.txt     ← Python dependencies
vercel.json          ← Vercel build config
.env.example         ← Environment template
docs/
  intents-catalog.md ← 20 intents with descriptions and conversation mapping
  jira/              ← Enriched ticket specs
```

### System prompt

The system prompt in `main.py` contains all the domain knowledge from the case study: clinic scope, drop-off/pick-up times, fasting rules, blood test policy, heat restrictions, transport requirements, post-op care, cancellation policy, and scheduling rules. This knowledge can also come from the RAG pipeline — both sources are compatible and may overlap.

### Conversation memory

Each `session_id` gets its own `InMemoryChatMessageHistory` via LangChain's `RunnableWithMessageHistory`. This means the bot remembers context within a session (e.g., if the user said "cat" in turn 1, it won't re-ask in turn 3).

---

## RAG pipeline (VET-11)

The RAG pipeline retrieves information from the official pre-operative instructions page.

**Source URL**: https://veterinary-clinic-teal.vercel.app/en/docs/instructions-before-operation

### How it works

1. **Fetch**: On startup, the app fetches the HTML from the official URL using `httpx`.
2. **Parse**: `BeautifulSoup` extracts clean text from the page (strips nav, scripts, styles).
3. **Chunk**: `RecursiveCharacterTextSplitter` splits the text into ~500-character chunks with 100-char overlap.
4. **Embed**: `OpenAIEmbeddings` converts each chunk into a vector.
5. **Index**: `FAISS` stores the vectors in an in-memory index.
6. **Retrieve**: For each user message, the top 4 most similar chunks are retrieved and injected into the system prompt as context.

### Verification

Check RAG status via the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "rag_loaded": true,
  "rag_chunks": 12,
  "rag_error": null
}
```

### Test questions for RAG

- "How long should my pet fast before surgery?" → Should mention 8–12 hours food, 1–2 hours water.
- "What should I bring on surgery day?" → Consent form, pet documentation, carrier for cats.
- "What do I do if my pet vomits after surgery?" → Normal on day of surgery, don't worry.

---

## Tool: check_availability (VET-12)

A mock tool that the LLM can invoke when the user asks about scheduling availability.

### What it does

The `check_availability` function returns a simulated weekly schedule with:

- **Surgery days**: Monday to Thursday only.
- **Daily budget**: 240 minutes maximum.
- **Dog limit**: Maximum 2 dogs per day.
- **Durations**: Dog ≈ 60 min, Cat ≈ 30 min.

### Mock data assumptions

The mock generates a schedule for the upcoming Mon–Thu with pre-set bookings:

| Day | Dogs booked | Cats booked | Minutes used | Can accept dog? | Can accept cat? |
|-----|-------------|-------------|--------------|-----------------|-----------------|
| Monday | 1 | 2 | 120/240 | Yes (1 slot) | Yes |
| Tuesday | 2 | 1 | 150/240 | No (limit) | Yes |
| Wednesday | 0 | 0 | 0/240 | Yes (2 slots) | Yes |
| Thursday | 1 | 0 | 60/240 | Yes (1 slot) | Yes |

### How the LLM invokes it

The tool is registered via OpenAI function calling (`bind_tools`). When the user asks about availability, the LLM decides to call `check_availability(species, preferred_day?)` and receives the schedule data, then formulates a natural-language response.

### Example invocation log

```
Tool call: check_availability({"species": "cat"})
→ Returns JSON with available days and capacity per day
```

---

## Intents (VET-5)

See [`docs/intents-catalog.md`](docs/intents-catalog.md) for the full catalog of 20 intents with descriptions, example utterances, expected behaviour, and a mapping from each of the 10 acceptance conversations to its primary intents.

---

## Vercel deployment (VET-3)

The project is configured for Vercel deployment:

- `vercel.json` routes all requests to `main.py` via `@vercel/python`.
- `api/index.py` re-exports the FastAPI app.
- Environment variables (`OPENAI_API_KEY`) must be set in the Vercel dashboard — never in the repo.

**Deploy URL**: https://enae-vet-es-weld.vercel.app


Steps to deploy:

1. Connect repo to Vercel (import from GitHub).
2. Set `OPENAI_API_KEY` in Vercel Environment Variables.
3. Deploy. The build uses `@vercel/python` with `requirements.txt`.
4. Verify: open the Vercel URL → chat UI should load, `/health` should return OK.

---

## Acceptance conversations

The chatbot is validated against 10 acceptance conversations defined in the course material:

| Conv. | Theme | Key test |
|-------|-------|----------|
| 1 | Greeting & scope | Welcomes; rejects general consults |
| 2 | Drop-off windows | Cat 08–09, dog 09–10:30; memory (species switch) |
| 3 | Blood test / age | Mandatory >6 years; recommended otherwise |
| 4 | Emergency | Redirects to emergency vet immediately |
| 5 | Heat restriction | Rejects spay for dog in heat; 2-month wait |
| 6 | Pick-up times | Dog ~12:00, cat ~15:00; memory (species switch) |
| 7 | Human handoff | Provides phone/WhatsApp escalation |
| 8 | Availability (tool) | Invokes tool; shows concrete days |
| 9 | Capacity (tool) | 2-dog limit; suggests alternatives |
| 10 | Pre-op fasting (RAG) | Fasting rules from RAG/prompt |

Conversations 1–7 = base 5 points (memory + domain, no tool).  
Conversations 8–9 = +1 point (tool de disponibilidad).  
Conversation 10 = +1 point (RAG pipeline).

---

## Docs overview

| Document | Contents |
|----------|----------|
| [`docs/intents-catalog.md`](docs/intents-catalog.md) | 20 intents + conversation mapping |
| [`docs/jira/`](docs/jira/) | Enriched ticket specs and examples |
