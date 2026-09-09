import sys
from pathlib import Path

import torch
import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

"""
Stage 2: Baseline RAG — retrieval + base model, NO fine-tuning, NO agent
it Loads the base model directly via transformers + 4-bit quantization for simplicity.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


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

If the reference material doesn't clearly cover the query, say so honestly \
rather than guessing — a wrong confident answer is worse than admitting \
uncertainty here.

Reference material:
{context}

Query: {query}
"""


class BaselineRAG:
    """Loads embedder, Qdrant client, and the base LLM once """

    def __init__(self, config: dict, use_fallback_model: bool = True):
        self.config = config
        vector_store_path = PROJECT_ROOT / config["paths"]["vector_store"]
        self.collection_name = config["vector_db"]["collection_name"]

        print("Loading embedder...")
        self.embedder = SentenceTransformer(config["embeddings"]["model"])

        print("Connecting to Qdrant...")
        self.qdrant = QdrantClient(path=str(vector_store_path))

        model_name = (
            config["model"]["fallback_model"]
            if use_fallback_model
            else config["model"]["base_model"]
        )
        print(f"Loading {model_name} in 4-bit on GPU (first run downloads the model, be patient)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        print("Model loaded.")

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        query_vec = self.embedder.encode(query).tolist()
        hits = self.qdrant.query_points(
            collection_name=self.collection_name, query=query_vec, limit=top_k
        ).points
        return [hit.payload["content"] for hit in hits]

    def generate(self, query: str, context_chunks: list[str]) -> str:
        prompt = FAULT_DIAGNOSIS_PROMPT.format(
            context="\n\n".join(context_chunks), query=query
        )
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        output = self.model.generate(
            **inputs, max_new_tokens=400, temperature=0.3, do_sample=True
        )
        response = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        return response

    def answer(self, query: str, top_k: int = 5) -> dict:
        chunks = self.retrieve(query, top_k=top_k)
        response = self.generate(query, chunks)
        return {"query": query, "retrieved_chunks": chunks, "response": response}


if __name__ == "__main__":
    config = load_config()
    rag = BaselineRAG(config, use_fallback_model=True)

    test_queries = [
        "Stop code E402 on Siemens S7-1200, motor won't start",
        "Communication module not working, what should I check?",
    ]
    for q in test_queries:
        print(f"\n{'=' * 60}\nQuery: {q}\n{'=' * 60}")
        result = rag.answer(q)
        print("Retrieved context:")
        for c in result["retrieved_chunks"][:3]:
            print(f"  - {c[:100]}")
        print(f"\nResponse:\n{result['response']}")

    rag.qdrant.close()