"""LangGraph version of the RAG pipeline: adds a routing decision instead of
always doing the same fixed retrieve-then-answer steps.

    START -> router -> "direct"  -> direct_answer -> END
                     -> "complex" -> retrieve -> respond -> END

The router here is a quick LLM classification call - a stand-in for the
fine-tuned classifier that replaces it on Day 5. The graph shape won't change
when that swap happens, only what runs inside the router node.
"""
import os
from typing import Literal, Optional, TypedDict

import chromadb
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "fastapi_docs"
TOP_K = 4

ROUTER_MODEL = "openai/gpt-oss-20b"   # small/fast: used for routing and the direct-answer path
RESPONDER_MODEL = "openai/gpt-oss-120b"  # larger: only invoked once a question is judged complex

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
_groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


class GraphState(TypedDict, total=False):
    question: str
    route: Literal["direct", "complex"]
    chunks: list[str]
    sources: list[str]
    answer: str


ROUTER_PROMPT = """Classify the following question about FastAPI into exactly one category:

- "direct": a general knowledge question answerable without looking up FastAPI's docs \
(e.g. definitions, high-level "what is X for" questions).
- "complex": requires specific FastAPI usage details, code syntax, or how-to steps that \
should be grounded in the actual documentation.

Respond with only the single word "direct" or "complex".

Question: {question}"""


def router_node(state: GraphState) -> dict:
    response = _groq_client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[{"role": "user", "content": ROUTER_PROMPT.format(question=state["question"])}],
        temperature=0,
    )
    label = response.choices[0].message.content.strip().lower()
    # default to the more thorough path if the classifier's output is unclear
    route: Literal["direct", "complex"] = "direct" if "direct" in label else "complex"
    return {"route": route}


def direct_answer_node(state: GraphState) -> dict:
    response = _groq_client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[{"role": "user", "content": state["question"]}],
    )
    return {"answer": response.choices[0].message.content, "sources": []}


def retrieve_node(state: GraphState) -> dict:
    collection = _chroma_client.get_collection(COLLECTION_NAME)
    query_embedding = _embed_model.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)
    chunks = results["documents"][0]
    sources = sorted({m["source"] for m in results["metadatas"][0]})
    return {"chunks": chunks, "sources": sources}


def respond_node(state: GraphState) -> dict:
    context = "\n\n---\n\n".join(state["chunks"])
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so plainly instead of guessing.

Context:
{context}

Question: {state["question"]}"""
    response = _groq_client.chat.completions.create(
        model=RESPONDER_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"answer": response.choices[0].message.content}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("router", router_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {"direct": "direct_answer", "complex": "retrieve"},
    )
    graph.add_edge("direct_answer", END)
    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


app = build_graph()


if __name__ == "__main__":
    question = input("Ask something about FastAPI: ")
    result = app.invoke({"question": question})

    print(f"\n--- Route: {result['route']} ---")
    print("\n--- Answer ---")
    print(result["answer"])
    if result.get("sources"):
        print("\n--- Sources ---")
        for s in result["sources"]:
            print(f"- {s}")
