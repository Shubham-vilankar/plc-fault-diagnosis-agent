import gc
import json
import re
import time
from pathlib import Path

import requests
import torch
import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

"""
Stage 6: Three-way evaluation ablation.

Compares three conditions:

  A. Base model (Qwen2.5-7B-Instruct, no fine-tuning) + RAG
  B. Fine-tuned model, NO RAG
  C. Fine-tuned model + RAG

Scoring -
The main way we check the model is:
- We compare the meaning of the model's answer with the correct answer.
- We use something called cosine similarity to measure how similar they are.
This is better than only checking whether the correct fault-code number appears.
We also check if the fault code matches, but this is only a secondary check.
Why? Because the model was trained to always write “Fault code {code}:”, 
so it could get the code right without actually giving the correct diagnosis.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
LOCAL_SERVER_URL = "http://localhost:8001/v1/chat/completions"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

FAULT_DIAGNOSIS_PROMPT = """You are an industrial fault-diagnosis assistant \
for PLC/SCADA systems. Given a stop code or fault description and the \
retrieved reference material below, respond in this structure:

1. Likely Cause
2. Confidence (low/medium/high)
3. Remedy Steps
4. Escalate to a technician? (yes/no, with reason)

If the reference material doesn't clearly cover the query, say so honestly \
rather than guessing.

Reference material:
{context}

Query: {query}
"""

NO_RAG_PROMPT = """You are an industrial fault-diagnosis assistant for \
PLC/SCADA systems. Respond in this structure:

1. Likely Cause
2. Confidence (low/medium/high)
3. Remedy Steps
4. Escalate to a technician? (yes/no, with reason)

Query: {query}
"""

CODE_EXTRACT_PATTERN = re.compile(r"[Ff]ault code (\S+)")


def load_eval_set() -> list[dict]:
    eval_path = PROJECT_ROOT / config["paths"]["sft_data"] / "eval.jsonl"
    examples = []
    with open(eval_path) as f:
        for line in f:
            ex = json.loads(line)
            user_msg = next(m["content"] for m in ex["messages"] if m["role"] == "user")
            expected_response = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
            match = CODE_EXTRACT_PATTERN.search(expected_response)
            expected_code = match.group(1).rstrip(":") if match else None
            examples.append({
                "query": user_msg,
                "expected_code": expected_code,
                "expected_response": expected_response,
            })
    return examples


def retrieve_context(embedder, qdrant, query: str, top_k: int = 5) -> str:
    query_vec = embedder.encode(query).tolist()
    hits = qdrant.query_points(
        collection_name=config["vector_db"]["collection_name"], query=query_vec, limit=top_k
    ).points
    return "\n\n".join(hit.payload["content"] for hit in hits)


def call_local_server(prompt: str, max_tokens: int = 400) -> tuple[str, float]:
    start = time.time()
    resp = requests.post(
        LOCAL_SERVER_URL,
        json={"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
        timeout=120,
    )
    latency = time.time() - start
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"], latency


def score_code_match(response: str, expected_code: str | None) -> bool:
    """WEAK secondary signal only — see module docstring. Does not measure
    diagnosis quality, only whether the code number appears."""
    if expected_code is None:
        return False
    return expected_code in response


def score_semantic_similarity(embedder, response: str, expected_response: str) -> float:
    """PRIMARY metric — cosine similarity between generated and ground-truth
    response embeddings. Measures whether the actual diagnosis content
    matches, robust to surface wording differences."""
    emb_response = embedder.encode(response, convert_to_tensor=True)
    emb_expected = embedder.encode(expected_response, convert_to_tensor=True)
    return float(util.cos_sim(emb_response, emb_expected).item())


def run_condition_b_and_c(examples: list[dict], embedder, qdrant) -> tuple[list[dict], list[dict]]:
    results_b, results_c = [], []
    for i, ex in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}] B+C: {ex['query'][:60]}...")

        prompt_b = NO_RAG_PROMPT.format(query=ex["query"])
        response_b, latency_b = call_local_server(prompt_b)
        results_b.append({
            "query": ex["query"], "expected_code": ex["expected_code"],
            "response": response_b, "latency": latency_b,
            "code_match": score_code_match(response_b, ex["expected_code"]),
            "semantic_similarity": score_semantic_similarity(embedder, response_b, ex["expected_response"]),
        })

        context = retrieve_context(embedder, qdrant, ex["query"])
        prompt_c = FAULT_DIAGNOSIS_PROMPT.format(context=context, query=ex["query"])
        response_c, latency_c = call_local_server(prompt_c)
        results_c.append({
            "query": ex["query"], "expected_code": ex["expected_code"],
            "response": response_c, "latency": latency_c,
            "code_match": score_code_match(response_c, ex["expected_code"]),
            "semantic_similarity": score_semantic_similarity(embedder, response_c, ex["expected_response"]),
        })
    return results_b, results_c


def run_condition_a(examples: list[dict], embedder, qdrant) -> list[dict]:
    print("Loading BASE model (no fine-tuning) for condition A...")
    model_name = config["model"]["fallback_model"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()

    results_a = []
    for i, ex in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}] A: {ex['query'][:60]}...")
        context = retrieve_context(embedder, qdrant, ex["query"])
        prompt = FAULT_DIAGNOSIS_PROMPT.format(context=context, query=ex["query"])

        start = time.time()
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=400, temperature=0.3, do_sample=True)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        latency = time.time() - start

        results_a.append({
            "query": ex["query"], "expected_code": ex["expected_code"],
            "response": response, "latency": latency,
            "code_match": score_code_match(response, ex["expected_code"]),
            "semantic_similarity": score_semantic_similarity(embedder, response, ex["expected_response"]),
        })

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Base model freed.")
    return results_a


def summarize(name: str, results: list[dict]) -> dict:
    n = len(results)
    avg_similarity = sum(r["semantic_similarity"] for r in results) / n if n else 0
    code_match_rate = sum(r["code_match"] for r in results) / n if n else 0
    avg_latency = sum(r["latency"] for r in results) / n if n else 0
    return {
        "condition": name, "n": n,
        "avg_semantic_similarity": avg_similarity,
        "code_match_rate": code_match_rate,
        "avg_latency_sec": avg_latency,
    }


if __name__ == "__main__":
    examples = load_eval_set()
    print(f"Loaded {len(examples)} eval examples\n")

    embedder = SentenceTransformer(config["embeddings"]["model"])
    qdrant = QdrantClient(path=str(PROJECT_ROOT / config["paths"]["vector_store"]))

    print("Running conditions B (fine-tuned, no RAG) and C (fine-tuned + RAG)...")
    results_b, results_c = run_condition_b_and_c(examples, embedder, qdrant)

    print("\nRunning condition A (base model + RAG)...")
    results_a = run_condition_a(examples, embedder, qdrant)

    qdrant.close()

    summary = [
        summarize("A: Base + RAG", results_a),
        summarize("B: Fine-tuned, no RAG", results_b),
        summarize("C: Fine-tuned + RAG", results_c),
    ]

    print(f"\n{'='*80}\nRESULTS\n{'='*80}")
    print(f"{'Condition':<25} {'N':>4} {'Sem. Similarity':>16} {'Code Match':>12} {'Avg Latency':>14}")
    for s in summary:
        print(f"{s['condition']:<25} {s['n']:>4} {s['avg_semantic_similarity']:>15.3f} "
              f"{s['code_match_rate']*100:>10.1f}% {s['avg_latency_sec']:>13.2f}s")
    print("\nNote: semantic similarity is the primary metric. Code match is a "
          "weak secondary signal (the fine-tuned model was trained to always "
          "echo the code number, so it's easy to satisfy without a correct "
          "diagnosis) — don't read code match alone as accuracy.")

    output_path = PROJECT_ROOT / "data" / "processed" / "ablation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "detailed": {"A": results_a, "B": results_b, "C": results_c},
        }, f, indent=2)
    print(f"\nFull results saved to {output_path}")