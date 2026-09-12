# Gatekeeper

An AI documentation assistant for FastAPI that **triages** incoming questions:
simple ones are answered fast and cheap, complex ones are escalated to a
larger model — the same "model cascading" pattern real companies use to keep
AI features fast and affordable at scale.

## Why this exists

Routing every question to the biggest available model is slow and expensive.
Gatekeeper uses a small, purpose-built classifier (fine-tuned specifically for
this) to decide, per question, whether it can be answered cheaply or needs to
be escalated — then verifies its own answer against the retrieved source docs
before returning it.

## Architecture

```
question -> [Router: fine-tuned small model] -> simple? -> answer directly
                                              -> complex? -> [Retriever (RAG)] -> [Groq LLM] -> [Critic] -> answer
```

(Diagram will be filled in as each piece is built — see `docs/architecture.md` from Day 7.)

See [LEARNING_LOG.md](LEARNING_LOG.md) for a day-by-day breakdown of what was
built and the concepts behind it.

## Status

- [x] Day 1 — repo scaffold + baseline RAG (fetch docs, embed, retrieve, ask)
- [ ] Day 2 — LangGraph orchestration
- [ ] Day 3 — tool use + self-verification (critic loop)
- [ ] Day 4 — fine-tuned router (LoRA)
- [ ] Day 5 — latency/cost instrumentation
- [ ] Day 6 — eval suite (accuracy/latency/cost comparison)
- [ ] Day 7 — deployment + dashboard + docs

## Setup

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add a free Groq API key from
https://console.groq.com/keys:

```bash
copy .env.example .env
```

## Day 1: baseline RAG

```bash
python src/fetch_docs.py   # pulls FastAPI's docs locally
python src/ingest.py       # chunks + embeds them into a local vector DB (Chroma)
python src/ask.py          # ask a question, get an answer grounded in the docs
```
