import json
import random
import sys
from pathlib import Path


"""
Stage 3a: Convert parsed FaultDocs into instruction/response pairs for QLoRA
fine-tuning.

IMPORTANT: docs are split into train/eval BEFORE generating query
phrasings, and shuffled at the doc level. This guarantees no fault code
appears in both train and eval in any form. An earlier version generated
phrasings first and split the flat list of examples afterward — that let
different wordings of the SAME fault code land on both sides, leaking the
"answer" into training in disguise and making eval accuracy look better
than real generalization.
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
]


def build_response(doc: FaultDoc) -> str:
    confidence = "medium" if doc.confidence == "high" else "low"
    escalate = "no" if doc.confidence == "high" else "yes"
    escalate_reason = (
        "Standard reference-documented fault, following the remedy steps should resolve it."
        if doc.confidence == "high"
        else "Source data for this fault was extracted with lower confidence — verify against the original manual before acting."
    )
    if doc.remedy:
        remedy_text = doc.remedy
    else:
        # Honest fallback ONLY if no remedy was captured in the source data — don't hallucinate a fix.
        remedy_text = (
            f"No specific corrective action was captured for this fault — "
            f"refer to fault code {doc.code} in the source documentation "
            f"({doc.metadata.get('file', 'reference manual')}) directly."
        )
    return (
        f"1. Likely Cause\n   {doc.content}\n\n"
        f"2. Confidence\n   {confidence}\n\n"
        f"3. Remedy Steps\n   {remedy_text}\n\n"
        f"4. Escalate to a technician? ({escalate})\n   {escalate_reason}"
    )


def build_sft_dataset(docs: list[FaultDoc], variations_per_doc: int = 2, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    for doc in docs:
        if not doc.code:
            continue
        templates = rng.sample(QUERY_TEMPLATES, k=min(variations_per_doc, len(QUERY_TEMPLATES)))
        response = build_response(doc)
        for template in templates:
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": template.format(code=doc.code)},
                    {"role": "assistant", "content": response},
                ]
            })
    rng.shuffle(examples)
    return examples


if __name__ == "__main__":
    raw_dir = PROJECT_ROOT / "data" / "raw"
    sft_dir = PROJECT_ROOT / "data" / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    docs = load_all(raw_dir)
    print(f"Loaded {len(docs)} fault docs")

    # Split by fault DOC first, then generate phrasings 

    random.seed(42)
    shuffled_docs = [d for d in docs if d.code]
    random.shuffle(shuffled_docs)
    split_idx = int(len(shuffled_docs) * 0.9)
    train_docs, eval_docs = shuffled_docs[:split_idx], shuffled_docs[split_idx:]

    train_examples = build_sft_dataset(train_docs, variations_per_doc=2)
    eval_examples = build_sft_dataset(eval_docs, variations_per_doc=2)

    train_path = sft_dir / "train.jsonl"
    eval_path = sft_dir / "eval.jsonl"

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(eval_path, "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    train_codes = {d.code for d in train_docs}
    eval_codes = {d.code for d in eval_docs}
    overlap = train_codes & eval_codes
    print(f"Wrote {len(train_examples)} train / {len(eval_examples)} eval examples")
    print(f"Train fault codes: {len(train_codes)}, Eval fault codes: {len(eval_codes)}, Overlap: {len(overlap)} (should be 0)")
    print(f"\nSample training example:\n{json.dumps(train_examples[0], indent=2)}")