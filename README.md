# Gatekeeper

An AI documentation assistant covering **FastAPI and LangGraph** that
**triages** incoming questions: a fine-tuned router decides which knowledge
base to search and whether the question is simple enough to answer cheaply
or needs escalation to a larger model — the same "model cascading" pattern
real companies use to keep AI features fast and affordable at scale. When
the answer involves code, the agent doesn't just explain it — it actually
**executes the generated snippet** in a sandbox and fixes it if it fails,
before ever showing it to the user.

## Why this exists

Routing every question to the biggest available model is slow and expensive.
Gatekeeper uses a small, purpose-built classifier (fine-tuned specifically for
this) to decide, per question, which docs to search and whether it can be
answered cheaply or needs to be escalated — then verifies its own answer,
running any generated code for real, before returning it.

## Architecture

```
question -> [Router: fine-tuned small model]
              -> decides domain (FastAPI / LangGraph / out-of-scope)
              -> decides complexity (simple / complex)
           -> simple -> answer directly
           -> complex -> [Retriever (RAG, domain-scoped)] -> [Groq LLM]
                       -> code involved? -> [Code executor] -> fails? -> retry
                       -> [Critic: checks answer against sources] -> answer
```

(Diagram will be filled in as each piece is built — see `docs/architecture.md` from Day 9.)

## Status

- [x] Day 1 — repo scaffold + baseline RAG (fetch docs, embed, retrieve, ask)
- [x] Day 2 — LangGraph orchestration (router -> direct-answer / retrieve+respond)
- [ ] Day 3 — multi-domain routing (add LangGraph docs as a second corpus)
- [ ] Day 4 — code-execution verification tool + retry loop
- [ ] Day 5 — fine-tuned router (LoRA, 5-way classifier)
- [ ] Day 6 — latency/cost instrumentation
- [ ] Day 7 — eval suite (accuracy/latency/cost comparison, incl. code-correctness)
- [ ] Day 8 — deployment + dashboard
- [ ] Day 9 — architecture diagram, README polish, demo recording, CI

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
