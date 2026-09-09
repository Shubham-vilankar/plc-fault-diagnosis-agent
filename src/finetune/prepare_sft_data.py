import json
import random
import sys
from pathlib import Path


"""
Stage 3a: Converting parsed FaultDocs into instruction/response pairs for QLoRA
fine-tuning.

With only ~265 real fault-code we are generating a few query phrasings per
fault code (different ways a technician might ask about the same fault), it just gives the model more examples of the
*response format and reasoning pattern* to learn from.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ingestion.load_documents import load_all, FaultDoc  # noqa: E402

SYSTEM_PROMPT = """You are an industrial fault-diagnosis assistant for PLC/SCADA systems. Given a stop code or fault description, respond in this structure:

1. Likely Cause
2. Confidence (low/medium/high)
3. Remedy Steps
4. Escalate to a technician? (yes/no, with reason)"""


QUERY_TEMPLATES = [
    "Stop code {code} on my PLC, what does it mean?",
    "Getting fault {code} — what should I check?",
    "What's causing error {code} and how do I fix it?",
    "PLC is showing {code}, need help diagnosing this.",
    "I see {code} on the HMI, what are the likely causes?",
    "Troubleshooting {code} — what steps should I take?",
    "My system threw {code}, what does that indicate?",
    "How do I resolve fault code {code}?",
    "how do i fix {code}?",
    "{code}",
    ]


def build_response(doc: FaultDoc) -> str:
    """
    this fucntion will Build a training-target response from the fault doc. Confidence and escalation are inferred conservatively from the source (high-confidence
    table extractions get 'medium' confidence responses; low-confidence regex-extracted ones get 'low' and always escalate) — this keeps the
    training targets honest rather than overclaiming certainty the source data doesn't support.
    """
    confidence = "medium" if doc.confidence == "high" else "low"
    escalate = "no" if doc.confidence == "high" else "yes"
    escalate_reason = (
        "Standard reference-documented fault, following the remedy steps should resolve it."
        if doc.confidence == "high"
        else "Source data for this fault was extracted with lower confidence — verify against the original manual before acting."
    )
    return (
        f"1. Likely Cause\n   {doc.content}\n\n"
        f"2. Confidence\n   {confidence}\n\n"
        f"3. Remedy Steps\n   Refer to fault code {doc.code} in the source documentation "
        f"({doc.metadata.get('file', 'reference manual')}) for the specific corrective action.\n\n"
        f"4. Escalate to a technician? ({escalate})\n   {escalate_reason}"
    )


def build_sft_dataset(docs: list[FaultDoc], variations_per_doc: int = 2, seed: int = 42) -> list[dict]:
    random.seed(seed)
    examples = []
    for doc in docs:
        if not doc.code:
            continue
        templates = random.sample(QUERY_TEMPLATES, k=min(variations_per_doc, len(QUERY_TEMPLATES)))
        response = build_response(doc)
        for template in templates:
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": template.format(code=doc.code)},
                    {"role": "assistant", "content": response},
                ]
            })
    random.shuffle(examples)
    return examples


if __name__ == "__main__":
    raw_dir = PROJECT_ROOT / "data" / "raw"
    sft_dir = PROJECT_ROOT / "data" / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    docs = load_all(raw_dir)
    print(f"Loaded {len(docs)} fault docs")

    examples = build_sft_dataset(docs, variations_per_doc=2)
    print(f"Generated {len(examples)} training examples")

    # 90/10 train/eval split
    split_idx = int(len(examples) * 0.9)
    train_examples, eval_examples = examples[:split_idx], examples[split_idx:]

    train_path = sft_dir / "train.jsonl"
    eval_path = sft_dir / "eval.jsonl"

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(eval_path, "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(train_examples)} train / {len(eval_examples)} eval examples")
    print(f"\nSample training example:\n{json.dumps(train_examples[0], indent=2)}")