from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Sequence

import time
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, default_data_collator


class InstructionDataset(Dataset):
    """Instruction tuning dataset using chat template formatting."""

    def __init__(self, records: Sequence[Dict[str, str]], tokenizer, max_length: int = 512):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.records[idx]

        prompt = f"{item['instruction']}\n\n{item['input']}"
        user_messages = [
            {"role": "user", "content": prompt},
        ]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": item["output"]},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        prompt_encoded = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_token_len = min(prompt_encoded["input_ids"].size(1), input_ids.size(0))

        labels = input_ids.clone()
        # Only optimize assistant answer tokens.
        labels[:prompt_token_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class LoraTrainingConfig:
    output_dir: str = "./qwen_lora_checkpoints"
    final_dir: str = "./qwen_lora_final"

    max_sequence_length: int = 1024   #512
    max_train_samples: Optional[int] = -1
    max_val_samples: Optional[int] = -1

    num_train_epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4  # Try a small grid: [1e-4, 2e-4]
    warmup_steps: int = 10
    logging_strategy: str = "epoch"
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 3

    use_fp16: bool = False
    use_bf16: bool = True
    optimizer_name: str = "adamw_torch"

    lora_rank: int = 8  #4
    lora_alpha: int = 16  #8
    lora_dropout: float = 0.3
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


def train_lora_model(
    base_model,
    tokenizer,
    train_records: Sequence[Dict[str, str]],
    val_records: Sequence[Dict[str, str]],
    config: Optional[LoraTrainingConfig] = None,
):
    """Run LoRA fine-tuning and save the final adapter model.

    Returns dict with trainer, model, timing, and output directory info.
    """
    if config is None:
        config = LoraTrainingConfig()

    # Optional sample truncation for quick experiments
    train_data = list(train_records)
    val_data = list(val_records)
    if config.max_train_samples not in (None, -1):
        train_data = train_data[: max(0, config.max_train_samples)]
    if config.max_val_samples not in (None, -1):
        val_data = val_data[: max(0, config.max_val_samples)]

    train_dataset = InstructionDataset(train_data, tokenizer, max_length=config.max_sequence_length)
    val_dataset = InstructionDataset(val_data, tokenizer, max_length=config.max_sequence_length)

    # --- Attach LoRA adapter ---
    peft_module = import_module("peft")
    LoraConfig = getattr(peft_module, "LoraConfig")
    TaskType = getattr(peft_module, "TaskType")
    get_peft_model = getattr(peft_module, "get_peft_model")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )
    model_for_training = get_peft_model(base_model, lora_config)

    # --- Build TrainingArguments ---
    if config.use_fp16 and config.use_bf16:
        raise ValueError("Only one of use_fp16 or use_bf16 can be True.")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=float(config.num_train_epochs),
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=config.use_fp16,
        bf16=config.use_bf16,
        logging_strategy=config.logging_strategy,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        warmup_steps=config.warmup_steps,
        report_to="none",
        optim=config.optimizer_name,
    )

    trainer = Trainer(
        model=model_for_training,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    start = time.time()
    trainer.train()
    train_seconds = time.time() - start

    trainer.save_model(config.final_dir)
    tokenizer.save_pretrained(config.final_dir)

    return {
        "trainer": trainer,
        "model": model_for_training,
        "train_seconds": train_seconds,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "output_dir": config.final_dir,
    }
