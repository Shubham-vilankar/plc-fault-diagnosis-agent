# PLC/SCADA Fault-Diagnosis Agent

An agentic RAG system for diagnosing industrial PLC/SCADA fault codes and stop
codes, combining retrieval-augmented generation, a QLoRA-fine-tuned local
model, and a CrewAI multi-agent pipeline — with full observability and a
local-GPU-vs-cloud-API cost/latency comparison.

## Why this project exists

Most "chat with your PDFs" GenAI portfolio projects look identical. This one
is grounded in real industrial automation domain knowledge (PLC/SCADA/HMI)
and is built to answer a hiring manager's actual questions: does the model
know the right answer, is fine-tuning worth the cost over RAG alone, and what
does it cost to run in production.

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

## Build stages

1. **Data collection & cleaning** — `src/ingestion/` — mixed sanitized
   company docs + public PLC/SCADA fault-code references.
2. **Baseline RAG** — `src/rag/` — retrieval + base model, no agent, no
   fine-tuning. This is the control group.
3. **QLoRA fine-tuning** — `src/finetune/` — Qwen2.5 fine-tuned on
   instruction/response pairs built from the fault-code data.
4. **CrewAI agent layer** — `src/agents/` — Retriever, Diagnostician,
   Escalator roles on top of the fine-tuned + RAG stack.
5. **Serving** — `src/serve/` — local model server (vLLM/llama.cpp) + AWS
   Lambda orchestration layer.
6. **Evaluation** — `src/eval/` — three-way ablation (base+RAG,
   fine-tuned-no-RAG, fine-tuned+RAG) on accuracy/faithfulness, latency, and
   cost-per-query (local GPU vs. Groq cloud baseline), traced in Langfuse.

## Status

- [ ] Data collection & cleaning
- [ ] Baseline RAG
- [ ] QLoRA fine-tuning
- [ ] CrewAI agent layer
- [ ] Serving (local + Lambda)
- [ ] Evaluation & cost comparison
- [ ] Write-up

## Hardware target

Local inference/fine-tuning on RTX 5080 (16GB VRAM). Qwen2.5-14B-Instruct at
4-bit fits comfortably; QLoRA fine-tuning uses the same quantized base with
LoRA adapters via `peft` + `trl` + `bitsandbytes`.
