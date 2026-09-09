"""
Stage 2: Baseline RAG — retrieval + base model, NO fine-tuning, NO agent.

This is your control group for the eventual three-way evaluation ablation
(base+RAG, fine-tuned-no-RAG, fine-tuned+RAG). Get this working end-to-end
FIRST before touching fine-tuning or CrewAI — it de-risks everything after it
and gives you a working demo early.

TODO (you):
  - Point EMBEDDING_MODEL / vector DB at your populated Qdrant collection
  - Swap the placeholder prompt for something tuned to fault-diagnosis output
    (structured: likely cause / confidence / remedy steps / when to escalate)
"""

import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


FAULT_DIAGNOSIS_PROMPT = """You are an industrial fault-diagnosis assistant \
for PLC/SCADA systems. Given a stop code or fault description and the \
retrieved reference material below, respond in this structure:

1. Likely Cause
2. Confidence (low/medium/high)
3. Remedy Steps
4. Escalate to a technician? (yes/no, with reason)

Reference material:
{context}

Query: {query}
"""


def retrieve(query: str, top_k: int = 5) -> list[str]:
    """
    Retrieve top_k relevant chunks from Qdrant for the query.
    TODO: qdrant_client search against the collection defined in config.yaml
    """
    raise NotImplementedError


def generate(query: str, context_chunks: list[str]) -> str:
    """
    Call the base (non-fine-tuned) local model with the fault-diagnosis
    prompt. TODO: wire up to your local vLLM/llama.cpp server (see
    src/serve/) or a HF `transformers` pipeline for early testing.
    """
    prompt = FAULT_DIAGNOSIS_PROMPT.format(
        context="\n\n".join(context_chunks), query=query
    )
    raise NotImplementedError


def answer(query: str) -> str:
    chunks = retrieve(query)
    return generate(query, chunks)


if __name__ == "__main__":
    test_query = "Stop code E402 on Siemens S7-1200, motor won't start"
    print(answer(test_query))
