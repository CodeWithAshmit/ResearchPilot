
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import process_pdf


DATA_DIR = Path("data")
DB_DIR = Path("chroma_db")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name="research_documents",
    metadata={"hnsw:space": "cosine"}
)


def index_pdf(pdf_path):
    print(f"\nProcessing: {pdf_path.name}")

    chunks = process_pdf(pdf_path)

    if not chunks:
        print("No text found in this PDF.")
        return

    # Remove the previous version of this PDF, if indexed.
    existing = collection.get(
        where={"source": pdf_path.name}
    )

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    texts = [chunk["text"] for chunk in chunks]

    print(f"Creating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    ids = [
        f"{pdf_path.name}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": pdf_path.name,
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(f"Successfully indexed {len(chunks)} chunks.")


def search_documents(question, top_k=3):
    total_chunks = collection.count()

    if total_chunks == 0:
        print("No documents indexed. Index a PDF first.")
        return []

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, total_chunks),
        include=["documents", "metadatas", "distances"]
    )

    matches = []

    for i, text in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # Convert cosine distance into cosine similarity.
        similarity = 1.0 - distance

        match = {
            "text": text,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": distance,
            "similarity": similarity
        }

        matches.append(match)

    return matches


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if pdf_files:
        print(f"Found {len(pdf_files)} PDF file(s).")

        for pdf_path in pdf_files:
            index_pdf(pdf_path)
    else:
        print("No PDFs found in the data folder.")

    print(f"\nTotal indexed chunks: {collection.count()}")

    while True:
        question = input(
            "\nAsk a question (or type 'exit' to quit): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        matches = search_documents(question)

        print("\n--- Retrieved passages ---")

        for i, match in enumerate(matches, start=1):
            print(f"\nResult {i}")
            print(f"Source: {match['source']}")
            print(f"Page: {match['page']}")
            print(
                f"Cosine distance: {match['distance']:.4f}"
            )
            print(
                f"Cosine similarity: {match['similarity']:.4f}"
            )
            print(match["text"][:700])
