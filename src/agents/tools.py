from pathlib import Path

import yaml
from crewai.tools import BaseTool
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

_embedder = SentenceTransformer(_config["embeddings"]["model"])
_qdrant = QdrantClient(path=str(PROJECT_ROOT / _config["paths"]["vector_store"]))
_collection = _config["vector_db"]["collection_name"]


class FaultCodeRetrieverTool(BaseTool):
    name: str = "fault_code_search"
    description: str = (
        "Search the PLC/SCADA fault-code knowledge base for entries relevant "
        "to a stop code, error code, or fault description. Returns the top "
        "matching fault-code entries with their descriptions."
    )

    def _run(self, query: str) -> str:
        query_vec = _embedder.encode(query).tolist()
        hits = _qdrant.query_points(
            collection_name=_collection, query=query_vec, limit=5
        ).points
        if not hits:
            return "No relevant fault-code entries found in the knowledge base."
        results = []
        for hit in hits:
            payload = hit.payload
            results.append(
                f"[score={hit.score:.2f}, confidence={payload.get('confidence')}] "
                f"{payload.get('content')}"
            )
        return "\n".join(results)