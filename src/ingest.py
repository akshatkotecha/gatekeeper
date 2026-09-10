"""
Day 1 - Step 2: turn the raw docs into searchable vectors.

Pipeline: read each .md file -> split into overlapping chunks -> embed each
chunk into a vector (a list of numbers capturing its meaning) -> store in
Chroma, a local vector database, so we can later find the chunks closest in
meaning to a user's question.
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("data/raw_docs")
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "fastapi_docs"

# Overlap matters: without it, a sentence that gets cut in half at a chunk
# boundary loses context. 150 chars of overlap between consecutive chunks
# keeps that from happening.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest():
    md_files = list(RAW_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit("No docs found in data/raw_docs. Run `python src/fetch_docs.py` first.")

    print("Loading embedding model (runs locally, no API needed)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # start clean each time this script runs, so re-ingesting doesn't duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, texts, metadatas = [], [], []
    for f in md_files:
        content = f.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(content)):
            if not chunk.strip():
                continue
            ids.append(f"{f.stem}_{i}")
            texts.append(chunk)
            metadatas.append({"source": f.name})

    print(f"Embedding {len(texts)} chunks from {len(md_files)} files...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"Done. Stored {len(texts)} chunks in Chroma at {CHROMA_DIR}/")


if __name__ == "__main__":
    ingest()
