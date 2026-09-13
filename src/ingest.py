"""Chunks and embeds data/raw_docs into a local Chroma vector store."""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("data/raw_docs")
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "fastapi_docs"

# overlap prevents context loss when a sentence is split across a chunk boundary
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

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # reset the collection so re-running this script doesn't duplicate chunks
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
