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
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": item["output"]},
        ]

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
        labels = input_ids.clone()
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

    max_sequence_length: int = 512
    max_train_samples: Optional[int] = -1
    max_val_samples: Optional[int] = -1

    num_train_epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    warmup_steps: int = 10
    logging_strategy: str = "epoch"
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 3

    use_fp16: bool = False
    use_bf16: bool = True
    optimizer_name: str = "adamw_torch"

    lora_rank: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.3
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


def select_records(records: Sequence[Dict[str, str]], max_samples: Optional[int]) -> List[Dict[str, str]]:
    """Select record subset for quick experiments."""
    data = list(records)
    if max_samples in (None, -1):
        return data
    return data[: max(0, max_samples)]


def attach_lora_adapter(model, config: LoraTrainingConfig):
    """Attach LoRA modules to the base CausalLM."""
    peft_module = import_module("peft")
    lora_config_cls = getattr(peft_module, "LoraConfig")
    task_type_cls = getattr(peft_module, "TaskType")
    get_peft_model_fn = getattr(peft_module, "get_peft_model")

    lora_config = lora_config_cls(
        task_type=task_type_cls.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )
    lora_model = get_peft_model_fn(model, lora_config)
    return lora_model


def build_training_arguments(config: LoraTrainingConfig) -> TrainingArguments:
    """Create TrainingArguments with transformers-version compatibility."""
    if config.use_fp16 and config.use_bf16:
        raise ValueError("Only one of use_fp16 or use_bf16 can be True.")

    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=float(config.num_train_epochs),
        per_device_train_batch_size=int(config.train_batch_size),
        per_device_eval_batch_size=int(config.eval_batch_size),
        gradient_accumulation_steps=int(config.gradient_accumulation_steps),
        learning_rate=float(config.learning_rate),
        fp16=bool(config.use_fp16),
        bf16=bool(config.use_bf16),
        logging_strategy=config.logging_strategy,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        save_total_limit=int(config.save_total_limit),
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        warmup_steps=int(config.warmup_steps),
        report_to="none",
        optim=config.optimizer_name,
    )


def train_lora_model(
    base_model,
    tokenizer,
    train_records: Sequence[Dict[str, str]],
    val_records: Sequence[Dict[str, str]],
    config: Optional[LoraTrainingConfig] = None,
):
    """Run LoRA fine-tuning and save the final adapter model."""
    if config is None:
        config = LoraTrainingConfig()

    train_subset = select_records(train_records, config.max_train_samples)
    val_subset = select_records(val_records, config.max_val_samples)

    train_dataset = InstructionDataset(train_subset, tokenizer, max_length=config.max_sequence_length)
    val_dataset = InstructionDataset(val_subset, tokenizer, max_length=config.max_sequence_length)

    model_for_training = attach_lora_adapter(base_model, config)
    training_args = build_training_arguments(config)

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
        "train_samples": len(train_subset),
        "val_samples": len(val_subset),
        "output_dir": config.final_dir,
    }
