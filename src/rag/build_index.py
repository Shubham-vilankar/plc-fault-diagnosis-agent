
import sys
from pathlib import Path

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

"""
Stage 1b: Embed the parsed FaultDoc chunks and load them into Qdrant.

Each FaultDoc is already one atomic unit (code + description + remedy kept
together during parsing) — so there's no further splitting. 
Using here Qdrant's local file-based mode (no
server/Docker required).
"""


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingestion.load_documents import load_all, FaultDoc  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_index(config: dict) -> None:
    raw_dir = PROJECT_ROOT / config["paths"]["raw_data"]
    vector_store_path = PROJECT_ROOT / config["paths"]["vector_store"]
    vector_store_path.mkdir(parents=True, exist_ok=True)

    print("Loading fault-code documents...")
    docs: list[FaultDoc] = load_all(raw_dir)
    print(f"  {len(docs)} documents loaded")

    print(f"Loading embedding model: {config['embeddings']['model']}...")
    embedder = SentenceTransformer(config["embeddings"]["model"])
    vector_size = embedder.get_sentence_embedding_dimension()

    print("Connecting to local Qdrant (file-based, no server needed)...")
    client = QdrantClient(path=str(vector_store_path))

    collection_name = config["vector_db"]["collection_name"]
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"Collection '{collection_name}' already exists — recreating with fresh data.")
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    print(f"Embedding {len(docs)} documents...")
    contents = [d.content for d in docs]
    embeddings = embedder.encode(contents, show_progress_bar=True, batch_size=32)

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "code": d.code,
                "title": d.title,
                "content": d.content,
                "source": d.source,
                "doc_type": d.doc_type,
                "confidence": d.confidence,
                **d.metadata,
            },
        )
        for i, d in enumerate(docs)
    ]

    print("Upserting into Qdrant...")
    client.upsert(collection_name=collection_name, points=points)

    count = client.count(collection_name=collection_name).count
    print(f"Done. {count} points indexed in collection '{collection_name}'.")

    # Sanity-check: run one real query and show what comes back
    test_query = "motor overload fault"
    query_vec = embedder.encode(test_query).tolist()
    hits = client.query_points(
        collection_name=collection_name, query=query_vec, limit=3
    ).points
    print(f"\nSanity check — top 3 results for '{test_query}':")
    for hit in hits:
        print(f"  score={hit.score:.3f} | {hit.payload['content'][:100]}")


if __name__ == "__main__":
    config = load_config()
    build_index(config)