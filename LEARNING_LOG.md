# Gatekeeper — Learning Log

This is your study companion for this project. Every day we add a section:
what we built, the concepts behind it in plain language, and the terms you
should be able to define cold in an interview. Read this instead of the code
comments when you want to actually understand *why*, not just *what*.

---

## The project, in one breath

**Gatekeeper** is an AI documentation assistant for FastAPI. Instead of
sending every question to one big, expensive, slow model, it **triages**
each question: a small model decides if it's simple enough to answer
cheaply, or if it needs to be escalated to a bigger model — the same
"model cascading" pattern real companies use to keep AI products fast and
affordable at scale. It also verifies its own answers against the source
docs before responding, and it's benchmarked with a real eval suite
(accuracy vs. latency vs. cost), not just vibes.

**Stack:** Python, LangGraph (orchestration), Chroma (vector DB), Groq
(free, fast LLM hosting), a self fine-tuned small model (the router),
FastAPI (serving), Streamlit (dashboard).

---

## Day 1 — Baseline RAG

### What we built
Three scripts, run in order:
1. `src/fetch_docs.py` — sparse-clones FastAPI's GitHub repo (just the
   `docs/en/docs` folder) so we have real, current documentation locally.
2. `src/ingest.py` — splits each doc into overlapping ~800-character
   chunks, turns each chunk into a vector (embedding) using a small local
   model (`all-MiniLM-L6-v2`), and stores them in **Chroma**, a local
   vector database.
3. `src/ask.py` — takes a question, embeds it the same way, finds the
   chunks whose vectors are closest in meaning, stuffs them into a prompt,
   and asks a Groq-hosted model to answer using only that context.

Result: ask "How do I define a path parameter with a default value in
FastAPI?" and get an answer grounded in the actual docs, with sources
listed — not the model guessing from stale training data.

### Concepts to know

**RAG (Retrieval-Augmented Generation)** — An LLM only knows what it was
trained on, and that training data has a cutoff. RAG fixes this by
fetching relevant real documents at question-time and handing them to the
model as context, so it answers from current, real material instead of
memory. This is the single most common pattern in real production AI
products (customer support bots, internal knowledge assistants, coding
copilots all use some form of this).

**Embedding** — A way of converting text into a list of numbers (a
vector) such that texts with similar *meaning* end up as similar-looking
vectors, even if they don't share exact words. "How do I set a default
path value?" and "assigning a default to a path param" would embed close
together even though barely any words match.

**Vector database** — A database built to answer "which of these
thousands of vectors are closest to this new vector?" quickly. Chroma is
a free, local one — good for learning and small projects. Production
systems often use a managed one (Pinecone, Weaviate) so it scales and
survives server restarts.

**Chunking** — *Why we don't just paste all 155 doc files into the
prompt:*
1. **Context window limits** — even large-context models have a cap, and
   FastAPI's full docs plus a system prompt could blow past it.
2. **Cost and latency** — you pay (and wait) per token sent to the model.
   Sending 100,000 irrelevant tokens to answer a one-line question is
   slow and wasteful.
3. **Precision** — stuffing in everything dilutes the signal. Retrieval
   finds the 3-4 chunks that actually answer the question, so the model's
   attention isn't split across irrelevant material — which measurably
   improves answer quality, not just cost.
   
   We chunk with **overlap** (150 characters) so a sentence that would
   otherwise get cut in half at a chunk boundary keeps its context in at
   least one of the two chunks.

**Groq** — An inference provider known for very low latency (they use
custom chips, not GPUs) and a free tier for open models. This project
uses Groq specifically because *latency* is one of the things we're going
to measure and optimize — it's not just "a free OpenAI alternative," it's
thematically the right tool.

### A live bug that's actually a feature (for your interview story)
When we tested Day 1, the model's answer slightly conflated *path*
parameters and *query* parameters — a small, believable-sounding
hallucination. This is the exact failure mode RAG alone doesn't fully
solve, and it's why Day 3 adds a **critic** step: an agent that
double-checks the draft answer against the retrieved chunks before it's
returned. Good talking point: *"I have a real example, from my own
build, of RAG producing a plausible-but-wrong answer — which is why I
added a verification step rather than trusting the first output."*

---

## What's coming (so you know the shape of the whole project)

- **Day 2 — LangGraph**: rebuilding this linear script as a graph with
  nodes and branching logic (Router → Retriever → Responder), so the
  system can make decisions instead of always doing the same fixed steps.
- **Day 3 — Tools + self-verification**: giving the agent the ability to
  call outside tools (e.g. web search), and adding the critic/verification
  loop mentioned above.
- **Day 4 — Fine-tuning**: LoRA fine-tuning a tiny open model (Qwen2.5-1.5B)
  to act as the router — deciding cheaply whether a question needs the
  big model at all.
- **Day 5 — Instrumentation**: measuring latency (p50/p95, not just
  average) and token cost per step of the pipeline.
- **Day 6 — Evals**: a 40-question golden test set, scoring accuracy,
  and a real comparison table: naive single-call baseline vs. full
  agentic pipeline, on accuracy/latency/cost.
- **Day 7 — Deploy**: shipping it (FastAPI + Streamlit, free hosting),
  writing the architecture doc, wiring evals into CI so a bad change
  can't silently ship.

---

## Glossary (quick interview review)

| Term | Plain-language definition |
|---|---|
| RAG | Fetching real documents at question-time instead of relying on the model's memory |
| Embedding | Turning text into numbers that capture meaning, so similar meanings end up as similar numbers |
| Vector DB | A database optimized for "find the closest vectors to this one" |
| Chunking | Splitting long documents into smaller pieces so retrieval is precise and cheap |
| Agent | A system where the LLM's output can influence *what happens next*, not just what's displayed |
| LangGraph | A framework for building agents as an explicit graph of steps (nodes) and decisions (edges), including loops |
| Model cascading | Routing easy requests to a cheap/fast model and only escalating hard ones to an expensive model |
| Fine-tuning | Further training an existing model on your own examples so it gets better at one specific, narrow task |
| LoRA | A cheap way to fine-tune: instead of updating all of a model's weights, you train small "adapter" layers, which is dramatically faster and cheaper |
| Latency (p50/p95) | p50 = the median response time; p95 = the time 95% of requests are faster than. p95 matters more in practice because it reflects your worst-common-case, not your best day |
| Eval / LLM-as-judge | A repeatable test suite that scores your AI system's outputs (often using another LLM to grade correctness) so you can prove quality instead of eyeballing it |

---

## Your interview elevator pitch (for reference)

> "I built an internal-knowledge-assistant pattern — the same
> architecture companies use for support/engineering copilots — and
> optimized specifically for what matters in production: not just answer
> quality, but cost and latency at scale. I fine-tuned a small model to
> handle routing so the expensive model is only invoked when actually
> needed, added a self-verification step after catching a real
> hallucination during testing, and built an eval harness to prove the
> tradeoffs quantitatively instead of eyeballing responses."
