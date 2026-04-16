# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Veterinary clinic chatbot and booking assistant (ENAE case study). A Python/LangChain/FastAPI backend that handles appointment scheduling, FAQs, and internal protocol queries for a vet clinic. The bot does not diagnose or prescribe -- it supports scheduling, FAQs, and internal procedures, citing tools or retrieved documents only.

**GitHub repo**: https://github.com/kuuli/enae-vet-es

## Tech Stack

- **Python** — backend language
- **LangChain** (and LangGraph) — orchestration: system prompts, tools, RAG pipelines, conversation memory
- **FastAPI** — HTTP/API layer
- **Poetry** — dependency management (preferred over Pipenv)
- **pytest** — testing framework (TDD encouraged)

## Architecture

The bot follows a four-layer design (defined in `.cursor/skills/langchain-vet-chatbots/SKILL.md`):

1. **System prompt** — single role per bot (e.g. reception, triage). Uses vet terminology: patient = animal, client = owner. Must never invent drugs, dosages, or procedures.
2. **Tools** — one LangChain tool per capability (`@tool` or `StructuredTool`, bound via `bind_tools`). Examples: `get_todays_appointments`, `search_patient_by_name`. Return structured data.
3. **RAG** — indexes clinic protocols, FAQs, and reference docs (not raw client/patient data). Semantic chunking; retrieve 3-5 top chunks; bot says "I don't have that information" when context is missing.
4. **Memory** — `ConversationBufferWindowMemory` or custom store, scoped per session/patient. Prompt template must include `chat_history`.

## Key Business Rules (Booking Flow)

These rules are defined in `docs/event-storming-workflow.md` and `docs/pre-operative-considerations.md`:

- **240-minute daily quota**: total occupied minutes + new appointment duration must not exceed 240 min
- **Dog limit**: max number of dogs per day
- **Drop-off windows**: cats 08:00-09:00, dogs 09:00-10:30
- **Surgical times are internal-only** — never shown to clients
- **Confirmation** includes: drop-off instructions + fasting protocol (last meal 8-12h before; water until 1-2h before)

## Project Structure

```
docs/                          # Business rules and domain docs (source of truth)
  event-storming-workflow.md   # Full booking flow, capacity rules, Mermaid diagram
  pre-operative-considerations.md  # Pre-op/post-op instructions (Spanish)
  jira/                        # Groomed Jira exports (enriched specs, before/after examples)
.cursor/
  agents/backend-langchain-vet.md  # Subagent definition for backend dev
  agents/product-manager.md        # Agent definition for ticket enrichment / grooming
  skills/langchain-vet-chatbots/   # LangChain skill guide + API reference
  skills/product-manager-ticket-enrichment/  # PM skill: phased Jira refinement
  commands/implement.md            # Jira ticket → PR workflow
  commands/enrich.md               # Jira ticket → phased PM enrichment workflow
  rules/                          # Cursor rules (Python best practices, repo context)
```

## Python Conventions

- PEP 8 with Black/Ruff formatting, 88-char line limit
- Always use type hints on all function signatures
- Imports: stdlib → third-party → local, sorted alphabetically (use isort)
- Docstrings: PEP 257, reStructuredText format
- `src/` layout for packages, `pyproject.toml` for config
- Use enums for symbolic constants, context managers for resources, f-strings for formatting
- Never use mutable default arguments or bare `except:`

## Veterinary Terminology (Use Consistently)

| Use             | Not                          |
|-----------------|------------------------------|
| patient         | pet, animal (formal context) |
| client          | owner, customer              |
| appointment     | booking, session             |
| record          | file, history                |
| protocol / SOP  | procedure, steps             |
| treatment       | meds, drugs                  |

## Development Workflow

The project uses a Jira-integrated workflow (`.cursor/commands/implement.md`): read ticket → plan from AC → develop via backend-langchain-vet agent → move ticket In Progress → open PR with AC checklist → move ticket to In Review.

Optional **upstream** grooming: `.cursor/commands/enrich.md` refines unclear tickets in phases using `.cursor/agents/product-manager.md` and `.cursor/skills/product-manager-ticket-enrichment/` before implementation.

Code and comments must be in **English**. Domain docs in `docs/` may be in Spanish.
