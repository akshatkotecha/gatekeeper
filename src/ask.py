"""Baseline RAG loop: retrieve relevant doc chunks, then answer via Groq."""
import os

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "fastapi_docs"
TOP_K = 4
GROQ_MODEL = "openai/gpt-oss-120b"


def retrieve(query: str, embed_model: SentenceTransformer, collection, k: int = TOP_K):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    return results["documents"][0], results["metadatas"][0]


def ask(query: str):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    chunks, metadatas = retrieve(query, embed_model, collection)
    context = "\n\n---\n\n".join(chunks)

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so plainly instead of guessing.

Context:
{context}

Question: {query}"""

    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content
    sources = sorted({m["source"] for m in metadatas})
    return answer, sources


if __name__ == "__main__":
    query = input("Ask something about FastAPI: ")
    answer, sources = ask(query)
    print("\n--- Answer ---")
    print(answer)
    print("\n--- Sources ---")
    for s in sources:
        print(f"- {s}")
