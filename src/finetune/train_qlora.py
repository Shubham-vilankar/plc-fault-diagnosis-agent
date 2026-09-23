"""
Stage 3b: QLoRA fine-tune Qwen2.5 on the fault-code SFT dataset.

Loads the base model in 4-bit (bitsandbytes), attaches LoRA adapters (peft),
and trains with trl's SFTTrainer. Only the small LoRA adapter is saved at
the end — a few tens of MB, not a full model copy — which is what gets
loaded back on top of the base model at inference time.

This is the longest-running step so far. Expect real wall-clock time even
on the RTX 5080, scaling with epochs/dataset size — this isn't something to
rerun casually, so double-check the config before kicking it off.
"""

import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main(use_fallback_model: bool = True):
    config = load_config()
    ft_cfg = config["finetuning"]
    sft_dir = PROJECT_ROOT / config["paths"]["sft_data"]
    adapter_out = PROJECT_ROOT / config["model"]["finetuned_adapter_path"]
    adapter_out.mkdir(parents=True, exist_ok=True)

    model_name = (
        config["model"]["fallback_model"]
        if use_fallback_model
        else config["model"]["base_model"]
    )
    print(f"Loading base model {model_name} in 4-bit for QLoRA training...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=ft_cfg["lora_r"],
        lora_alpha=ft_cfg["lora_alpha"],
        lora_dropout=ft_cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        # Standard target modules for Qwen2-family attention + MLP blocks
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    print("Loading SFT dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(sft_dir / "train.jsonl"),
            "eval": str(sft_dir / "eval.jsonl"),
        },
    )
    print(f"  train: {len(dataset['train'])}  eval: {len(dataset['eval'])}")

    training_args = SFTConfig(
        output_dir=str(adapter_out),
        num_train_epochs=ft_cfg["epochs"],
        per_device_train_batch_size=ft_cfg["batch_size"],
        gradient_accumulation_steps=ft_cfg["gradient_accumulation_steps"],
        learning_rate=float(ft_cfg["learning_rate"]),
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        max_length=1024,
        report_to="none",  # switch to "langfuse" or wire up manually once observability stage is set up
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {adapter_out}...")
    trainer.save_model(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))
    print("Done.")


if __name__ == "__main__":
    main(use_fallback_model=True)