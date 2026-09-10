"""
Local OpenAI-compatible chat completions server, backed by the same
transformers-based loading used in baseline_rag.py (not vLLM — sidesteps
the vLLM V2 Model Runner / WSL2 UVA incompatibility hit during setup).

This is a pragmatic stand-in for the agent-building stage. Revisit vLLM (or
an updated version / different flags) at the actual production-serving
stage — this server's job right now is just to give CrewAI something to
talk to that speaks the standard chat-completions API shape.

Run with: uvicorn src.serve.local_llm_server:app --port 8001
"""

import time
import uuid
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["fallback_model"]
ADAPTER_PATH = PROJECT_ROOT / config["model"]["finetuned_adapter_path"]

print(f"Loading base model {MODEL_NAME} in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto"
)

if ADAPTER_PATH.exists():
    print(f"Attaching LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
else:
    print("No adapter found, serving base model.")
    model = base_model

model.eval()
print("Model ready.")

app = FastAPI(title="Local fault-diagnosis LLM server")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "fault-diagnosis"
    messages: list[ChatMessage]
    max_tokens: int = 400
    temperature: float = 0.3


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
        )
    response_text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )

    # Standard OpenAI chat-completions response shape — this is what makes
    # CrewAI (and anything else expecting an OpenAI-compatible API) work
    # against this server without any special-casing.
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[-1],
            "completion_tokens": output.shape[-1] - inputs["input_ids"].shape[-1],
            "total_tokens": output.shape[-1],
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "adapter_loaded": ADAPTER_PATH.exists()}