
# PLC/SCADA Fault-Diagnosis Agent

I built this project to solve a real problem: PLC/HMI fault codes are often buried in huge vendor manuals, making troubleshooting slow.

This AI assistant reads manuals, finds the relevant information, and gives a structured diagnosis and solution. If it doesn't know the answer, it says "I don't know" instead of guessing.

It's built specifically for PLC/SCADA fault diagnosis using real Siemens, Allen-Bradley, and Mitsubishi documentation, and I also tested and evaluated its performance rather than just creating a demo.

## What it actually does

You give it a fault code or a description of what's going wrong — something
like "Stop code E402 on a Siemens S7-1200, motor won't start" — and it:

1. Searches a knowledge base built from real PLC/SCADA manuals for relevant
   fault-code entries
2. Reasons through a diagnosis: likely cause, confidence level, remedy steps
3. Decides whether the situation needs a human technician, and says why
4. Refuses to answer, honestly, if the question isn't something the
   knowledge base actually covers — instead of making something up

## Why it's built the way it is

Fine-tuning alone is not enough. I tested this myself instead of just assuming it.
At the beginning, I fine-tuned a model using the fault-code data. I thought fine-tuning alone would be enough to give good diagnoses. It wasn't.
This does not mean fine-tuning is bad. The problem is that fine-tuning mainly teaches the model how to answer, not facts that were never included in its training data.

Without retrieval, the model could correctly give a fault-code number because it learned the pattern. But it could still give the wrong cause or solution, because it did not have the actual information it needed. It was basically guessing based on patterns.
To prove this, I tested three versions using the same test data:

[ Base model + Retrieval ]

[ Fine-tuned model without Retrieval ]

[ Fine-tuned model + Retrieval ]

I kept everything else the same so the comparison was fair.The results showed that the fine-tuned model without retrieval was slightly worse at giving the correct diagnosis than the base model with retrieval.

The best results came from using both:

---Fine-tuning → teaches the model how to respond

---Retrieval (RAG) → gives the model the correct facts

That's why the final system uses both fine-tuning and RAG.

**Why the agents are separate from the evaluation**
The project also has a CrewAI multi-agent system with three agents:

Retriever → finds the relevant information
Diagnostician → figures out the likely problem
Escalation Reviewer → reviews the diagnosis and decides if it needs further attention

These agents are not included in the main comparison on purpose.

If I included the agents in the comparison, it would be difficult to know whether the improvement came from:

fine-tuning, retrieval, or the multi-agent system.

So tested each important part separately. This makes the results more fair and easier to understand.

**Model and setup**

The model used is Qwen2.5-7B-Instruct.It was fine-tuned using QLoRA on around 500 fault-diagnosis examples.The model runs locally on an RTX 5080 GPU and is quantized so it can comfortably run within 16 GB of VRAM.
A small FastAPI server connects the model to CrewAI using the API format CrewAI expects.



## Architecture

```
                       ┌─────────────────────┐
                       │   Knowledge Base     │
                       │ (fault codes, SOPs,  │
                       │  manuals) → Qdrant   │
                       └──────────┬───────────┘
                                  │ retrieval
┌───────────┐   query    ┌───────▼────────┐    ┌──────────────┐
│  Lambda    ├───────────►  CrewAI Crew    ├───►│ Local LLM     │
│ (API layer)│            │ - Retriever    │    │ Qwen2.5-14B   │
└───────────┘            │ - Diagnostician│    │ QLoRA + 4-bit │
                          │ - Escalator    │    │ (RTX 5080)    │
                          └───────┬────────┘    └──────────────┘
                                  │ traces
                          ┌───────▼────────┐
                          │    Langfuse     │
                          │ (self-hosted)   │
                          └────────────────┘
```

```
Fault manuals (PDF) --> parsed into fault-code entries --> embedded and
indexed in Qdrant (local, no server needed)

Query --> guardrails (input validation, scope check) -->
  CrewAI: Retriever agent (searches Qdrant)
       -> Diagnostician agent (reasons over retrieved context, fine-tuned model)
       -> Escalation Reviewer agent (decides if a human needs to step in)
  --> guardrails (output validation) --> final answer
```

## What's actually in this repo

- `src/ingestion/` — parses PLC/SCADA manuals into structured fault-code
  entries. Not trivial: real manuals have inconsistent table formats, OCR
  artifacts, and tables that look like fault codes but aren't.
- `src/rag/` — retrieval + baseline single-shot generation. This is the
  control group everything else is compared against.
- `src/finetune/` — QLoRA fine-tuning, including the SFT data prep. The
  train/eval split here is deliberately done *before* generating question
  phrasings, so a fault code never appears in both sets.
- `src/agents/` — the CrewAI multi-agent layer, plus the guardrail module
  (input validation, prompt-injection pattern detection, retrieval-score
  scope checking, and output structure validation).
- `src/eval/` — the three-way ablation script. Scores on semantic
  similarity to the ground-truth response (not just whether a fault-code
  number happens to appear — that check alone is trivially gameable, since
  the fine-tuned model was trained to always echo the code).
- `src/serve/` — the local model server.

## Results

Three-way comparison on 53 held-out fault-diagnosis questions (fault codes
never seen during fine-tuning), scored by semantic similarity to the
correct diagnosis:

| Condition | Semantic similarity | Code mentioned | Avg latency |
|---|---|---|---|
| Base model + RAG | 0.839 | 90.6% | 4.85s |
| Fine-tuned, no RAG | 0.831 | 100.0% | 7.44s |
| Fine-tuned + RAG | **0.898** | 100.0% | 6.09s |

The "code mentioned" column is included for transparency but isn't the
headline number — it's easy to satisfy by design (see above) and doesn't by
itself mean the diagnosis was right. Semantic similarity is the metric that
actually reflects content quality.


## Running it locally
See `configs/config.yaml` for all tunable settings.

1. `src/ingestion/load_documents.py` — parse manuals into fault docs
2. `src/rag/build_index.py` — embed and index into local Qdrant
3. `src/finetune/prepare_sft_data.py` then `train_qlora.py` — fine-tune
4. `src/serve/local_llm_server.py` — serve the fine-tuned model
5. `src/agents/crew.py` — run the full agent pipeline
6. `src/eval/run_ablation.py` — reproduce the evaluation results above

Built and tested on Python 3.11, WSL2 Ubuntu, RTX 5080 (16GB VRAM).

## Hardware target

Local inference/fine-tuning on RTX 5080 (16GB VRAM). Qwen2.5-14B-Instruct at
4-bit fits comfortably; QLoRA fine-tuning uses the same quantized base with
LoRA adapters via `peft` + `trl` + `bitsandbytes`.