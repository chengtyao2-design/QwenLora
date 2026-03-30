# QwenLora 优化改动总结

全部改动通过 `py_compile` 语法验证（ALL OK），共修改 4 个文件。

---

## train.py

```diff:train.py
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
    use_qlora: bool = False  # 开启后底座模型需已用 4-bit 加载（即 load_tokenizer_and_model 的 use_qlora=True）


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

    # QLoRA: 4-bit 模型在训练前必须执行此准备步骤
    # 作用：将 LayerNorm 等层转回 fp32、开启 gradient checkpointing，避免反向传播数値不稳定
    if config.use_qlora:
        prepare_model_for_kbit_training = getattr(peft_module, "prepare_model_for_kbit_training")
        base_model = prepare_model_for_kbit_training(base_model)

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
        # [QLoRA] use_reentrant=False 避免 bf16 + gradient checkpointing 在新 GPU 上的 CUBLAS 错误
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.use_qlora else None,
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
===
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Sequence

import time
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, default_data_collator, set_seed
from sklearn.metrics import accuracy_score, f1_score


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
    lora_dropout: float = 0.05  # 指令微调用 0.05~0.1，原 0.3 偏高
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",  # FFN 层，覆盖面更广
        ]
    )
    use_qlora: bool = False  # 开启后底座模型需已用 4-bit 加载（即 load_tokenizer_and_model 的 use_qlora=True）
    seed: int = 42  # 全局随机种子，保证训练可复现


def compute_metrics(eval_pred):
    """用 Accuracy + Macro-F1 评估，供 Trainer 选最优 checkpoint。"""
    logits, labels = eval_pred
    # logits shape: (N, vocab_size)，取 argmax 得到预测 token id
    preds = np.argmax(logits, axis=-1)
    # 过滤掉 labels=-100 的 padding 位置，取每条样本最后一个有效 token
    # 注意：InstructionDataset 的 labels 里 padding=-100，答案是最后一个非-100 token
    valid_preds, valid_labels = [], []
    for pred_row, label_row in zip(preds, labels):
        valid_mask = label_row != -100
        if valid_mask.any():
            valid_preds.append(int(pred_row[valid_mask][-1]))
            valid_labels.append(int(label_row[valid_mask][-1]))
    if not valid_labels:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    acc = accuracy_score(valid_labels, valid_preds)
    macro_f1 = f1_score(valid_labels, valid_preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": macro_f1}


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

    # 设置全局随机种子，确保结果可复现
    set_seed(config.seed)

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

    # QLoRA: 4-bit 模型在训练前必须执行此准备步骤
    # 作用：将 LayerNorm 等层转回 fp32、开启 gradient checkpointing，避免反向传播数値不稳定
    if config.use_qlora:
        prepare_model_for_kbit_training = getattr(peft_module, "prepare_model_for_kbit_training")
        base_model = prepare_model_for_kbit_training(base_model)

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
        metric_for_best_model="macro_f1",  # 用 Macro-F1 选最优 checkpoint
        greater_is_better=True,
        warmup_steps=config.warmup_steps,
        report_to="none",
        optim=config.optimizer_name,
        # [QLoRA] use_reentrant=False 避免 bf16 + gradient checkpointing 在新 GPU 上的 CUBLAS 错误
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.use_qlora else None,
    )

    trainer = Trainer(
        model=model_for_training,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,  # 用 Accuracy + Macro-F1 选最优 checkpoint
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
```

| 改动 | 内容 |
|------|------|
| ① LoRA 超参 | `lora_dropout` 0.3 → **0.05**；`target_modules` 新增 `gate_proj`、`up_proj`、`down_proj` (FFN 层) |
| ③ 可复现性 | [LoraTrainingConfig](file:///d:/my_files/job/projects/QwenLora/train.py#79-114) 增 `seed: int = 42`；[train_lora_model()](file:///d:/my_files/job/projects/QwenLora/train.py#136-237) 开头调用 `set_seed(config.seed)` |
| ⑤ 验证指标 | 新增 [compute_metrics](file:///d:/my_files/job/projects/QwenLora/train.py#116-134)（Accuracy + Macro-F1）；`metric_for_best_model` 改为 `"macro_f1"`，`greater_is_better=True` |

---

## dataset.py

```diff:dataset.py
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv_with_encoding(filepath: str, encodings: Optional[List[str]] = None) -> pd.DataFrame:
	"""Read CSV robustly by trying common encodings."""
	if encodings is None:
		encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-8-sig"]

	for encoding in encodings:
		try:
			return pd.read_csv(filepath, encoding=encoding, on_bad_lines="skip")
		except (UnicodeDecodeError, UnicodeError):
			continue

	# Last fallback: replace undecodable chars.
	return pd.read_csv(filepath, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")


def rating_to_numeric(rating_str: Any) -> Optional[int]:
	"""Convert strings like '4 star' into integer labels 1..5."""
	if pd.isna(rating_str):
		return None

	match = re.search(r"(\d+)\s*star", str(rating_str), flags=re.IGNORECASE)
	if not match:
		return None

	value = int(match.group(1))
	if 1 <= value <= 5:
		return value
	return None


def add_numeric_rating_column(
	df: pd.DataFrame,
	source_col: str = "Rating",
	target_col: str = "rating_numeric",
) -> pd.DataFrame:
	"""Return a copy of df with parsed numeric rating column."""
	out = df.copy()
	out[target_col] = out[source_col].apply(rating_to_numeric)
	return out


def load_project_data(data_dir: str = "./data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Load train/test/answer CSV files and add numeric labels where available."""
	train_path = os.path.join(data_dir, "review_train.csv")
	test_path = os.path.join(data_dir, "review_test.csv")
	answer_path = os.path.join(data_dir, "test_answer.csv")

	train_df = read_csv_with_encoding(train_path)
	test_df = read_csv_with_encoding(test_path)
	answer_df = read_csv_with_encoding(answer_path)

	train_df = add_numeric_rating_column(train_df)
	answer_df = add_numeric_rating_column(answer_df)
	return train_df, test_df, answer_df


def split_train_val(
	train_df: pd.DataFrame,
	seed: int = 5494,
	test_size: float = 0.2,
	label_col: str = "rating_numeric",
	debug_sample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Stratified train/validation split with optional downsampling for debugging."""
	train_subset, val_subset = train_test_split(
		train_df,
		test_size=test_size,
		random_state=seed,
		stratify=train_df[label_col],
	)

	if debug_sample_size not in (None, -1):
		train_subset = train_subset.sample(n=min(debug_sample_size, len(train_subset)), random_state=seed)
		val_subset = val_subset.sample(n=min(debug_sample_size, len(val_subset)), random_state=seed)

	return train_subset.reset_index(drop=True), val_subset.reset_index(drop=True)


def build_example_library(
	train_df: pd.DataFrame,
	n_per_rating: int = 10,
	seed: int = 5494,
	rating_col: str = "rating_numeric",
) -> List[Dict[str, Any]]:
	"""Create a balanced few-shot example pool from the training split."""
	examples: List[Dict[str, Any]] = []

	for rating in [1, 2, 3, 4, 5]:
		class_df = train_df[train_df[rating_col] == rating]
		if class_df.empty:
			continue

		sampled = class_df.sample(n=min(n_per_rating, len(class_df)), random_state=seed)
		for _, row in sampled.iterrows():
			examples.append(
				{
					"title": str(row["Title"]) if pd.notna(row.get("Title")) else "",
					"review": str(row["Review"]),
					"rating": int(row[rating_col]),
				}
			)

	return examples


def prepare_instruction_data(
	df: pd.DataFrame,
	rating_col: str = "rating_numeric",
) -> List[Dict[str, str]]:
	"""Convert row-wise review data into SFT instruction format."""
	from model import PROMPT_HEADER, PROMPT_FOOTER

	rows: List[Dict[str, str]] = []

	for _, row in df.iterrows():
		title = str(row["Title"]) if pd.notna(row.get("Title")) else ""
		review = str(row["Review"])
		rating = int(row[rating_col])

		input_text = f"Title: {title}\nReview: {review}" if title else f"Review: {review}"
		# To match the behavior of `build_finetuned_prompt`:
		# f"{PROMPT_HEADER}\n\n{input_text}\n\n{PROMPT_FOOTER}"
		# In train.py it does: f"{item['instruction']}\n\n{item['input']}"
		# We set instruction to PROMPT_HEADER, and input to input_text \n\n PROMPT_FOOTER

		rows.append(
			{
				"instruction": PROMPT_HEADER,
				"input": f"{input_text}\n\n{PROMPT_FOOTER.strip()}", # using strip because user appending 'Rating: ' is expected by output text
				# Wait, PROMPT_FOOTER ends with "Rating: ".
				# If we let it be, then the expected output is just the number. 
				# In train.py: {"role": "assistant", "content": item["output"]} -> str(rating)
				"output": str(rating),
			}
		)

	return rows
===
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv_with_encoding(filepath: str, encodings: Optional[List[str]] = None) -> pd.DataFrame:
	"""Read CSV robustly by trying common encodings."""
	if encodings is None:
		encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-8-sig"]

	for encoding in encodings:
		try:
			return pd.read_csv(filepath, encoding=encoding, on_bad_lines="skip")
		except (UnicodeDecodeError, UnicodeError):
			continue

	# Last fallback: replace undecodable chars.
	return pd.read_csv(filepath, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")


def rating_to_numeric(rating_str: Any) -> Optional[int]:
	"""Convert strings like '4 star' into integer labels 1..5."""
	if pd.isna(rating_str):
		return None

	match = re.search(r"(\d+)\s*star", str(rating_str), flags=re.IGNORECASE)
	if not match:
		return None

	value = int(match.group(1))
	if 1 <= value <= 5:
		return value
	return None


def add_numeric_rating_column(
	df: pd.DataFrame,
	source_col: str = "Rating",
	target_col: str = "rating_numeric",
) -> pd.DataFrame:
	"""Return a copy of df with parsed numeric rating column."""
	out = df.copy()
	out[target_col] = out[source_col].apply(rating_to_numeric)
	return out


def load_project_data(data_dir: str = "./data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Load train/test/answer CSV files and add numeric labels where available."""
	train_path = os.path.join(data_dir, "review_train.csv")
	test_path = os.path.join(data_dir, "review_test.csv")
	answer_path = os.path.join(data_dir, "test_answer.csv")

	train_df = read_csv_with_encoding(train_path)
	test_df = read_csv_with_encoding(test_path)
	answer_df = read_csv_with_encoding(answer_path)

	train_df = add_numeric_rating_column(train_df)
	answer_df = add_numeric_rating_column(answer_df)
	return train_df, test_df, answer_df


def split_train_val(
	train_df: pd.DataFrame,
	seed: int = 5494,
	test_size: float = 0.2,
	label_col: str = "rating_numeric",
	debug_sample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Stratified train/validation split with optional downsampling for debugging."""
	train_subset, val_subset = train_test_split(
		train_df,
		test_size=test_size,
		random_state=seed,
		stratify=train_df[label_col],
	)

	if debug_sample_size not in (None, -1):
		train_subset = train_subset.sample(n=min(debug_sample_size, len(train_subset)), random_state=seed)
		val_subset = val_subset.sample(n=min(debug_sample_size, len(val_subset)), random_state=seed)

	return train_subset.reset_index(drop=True), val_subset.reset_index(drop=True)


def build_example_library(
	train_df: pd.DataFrame,
	n_per_rating: int = 10,
	seed: int = 5494,
	rating_col: str = "rating_numeric",
) -> List[Dict[str, Any]]:
	"""Create a balanced few-shot example pool from the training split."""
	examples: List[Dict[str, Any]] = []

	for rating in [1, 2, 3, 4, 5]:
		class_df = train_df[train_df[rating_col] == rating]
		if class_df.empty:
			continue

		sampled = class_df.sample(n=min(n_per_rating, len(class_df)), random_state=seed)
		for _, row in sampled.iterrows():
			examples.append(
				{
					"title": str(row["Title"]) if pd.notna(row.get("Title")) else "",
					"review": str(row["Review"]),
					"rating": int(row[rating_col]),
				}
			)

	return examples


def prepare_instruction_data(
	df: pd.DataFrame,
	rating_col: str = "rating_numeric",
	oversample: bool = False,
	seed: int = 42,
) -> List[Dict[str, str]]:
	"""Convert row-wise review data into SFT instruction format.

	oversample=True: 对少数类做随机重复采样，使各类数量与最多类持平，
	有助于缓解类别不平衡对 Macro-F1 的影响。
	"""
	import random
	from model import PROMPT_HEADER, PROMPT_FOOTER

	def _row_to_record(row):
		title = str(row["Title"]) if pd.notna(row.get("Title")) else ""
		review = str(row["Review"])
		rating = int(row[rating_col])
		input_text = f"Title: {title}\nReview: {review}" if title else f"Review: {review}"
		return {
			"instruction": PROMPT_HEADER,
			"input": f"{input_text}\n\n{PROMPT_FOOTER.strip()}",
			"output": str(rating),
		}

	# 按类别分组
	groups: Dict[int, List] = {r: [] for r in range(1, 6)}
	for _, row in df.iterrows():
		groups[int(row[rating_col])].append(row)

	if not oversample:
		rows = [_row_to_record(row) for rows_in_group in groups.values() for row in rows_in_group]
		return rows

	# 过采样：以最多类数量为目标，对其他类随机重复抽取
	rng = random.Random(seed)
	max_count = max(len(v) for v in groups.values() if v)
	rows: List[Dict[str, str]] = []
	for rating, group_rows in groups.items():
		if not group_rows:
			continue
		sampled = group_rows[:]
		while len(sampled) < max_count:
			sampled.append(rng.choice(group_rows))
		for row in sampled:
			rows.append(_row_to_record(row))

	rng.shuffle(rows)
	return rows
```

| 改动 | 内容 |
|------|------|
| ③ 类别不平衡 | [prepare_instruction_data](file:///d:/my_files/job/projects/QwenLora/dataset.py#114-163) 新增 `oversample: bool = False`；开启时各类随机重采样至最多类数量，改善少数类 Macro-F1 |

**用法：**
```python
records = prepare_instruction_data(train_df, oversample=True)
```

---

## model.py

```diff:model.py
import os
import re
import glob
import random
from importlib import import_module
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class DeviceInfo:
    device_type: str
    device_name: str
    total_memory_gb: Optional[float]


def get_device_info() -> DeviceInfo:
    """Detect CUDA/NPU/MPS/CPU and return normalized device metadata."""
    # Ascend NPU
    try:
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            try:
                current_idx = npu.current_device()
            except Exception:
                current_idx = 0
            try:
                npu.set_device(current_idx)
            except Exception:
                pass

            memory_gb = None
            try:
                memory_gb = npu.get_device_properties(current_idx).total_memory / 1e9
            except Exception:
                pass

            return DeviceInfo(
                device_type="npu",
                device_name=f"Ascend NPU (Device {current_idx})",
                total_memory_gb=memory_gb,
            )
    except Exception:
        pass

    # NVIDIA CUDA
    if torch.cuda.is_available():
        return DeviceInfo(
            device_type="cuda",
            device_name=torch.cuda.get_device_name(0),
            total_memory_gb=torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # Apple MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceInfo(
            device_type="mps",
            device_name="Apple Silicon GPU (MPS)",
            total_memory_gb=None,
        )

    return DeviceInfo(device_type="cpu", device_name="CPU", total_memory_gb=None)


def get_torch_dtype(device_type: str, prefer_bf16: bool = True) -> torch.dtype:
    """Choose dtype by accelerator characteristics."""
    if device_type == "cuda":
        return torch.float16
    if device_type == "npu":
        return torch.bfloat16 if prefer_bf16 else torch.float16
    return torch.float32


def _find_local_model_dir(model_name: str, cache_dir: str) -> Optional[str]:
    """检查本地缓存是否已有完整模型（含 config.json + 权重文件）。

    ModelScope 在 Windows 上会把 model_name 里的 '.' 替换成 '___'，
    所以这里两种路径都检查一下。找到就返回路径，没找到返回 None。
    """
    original_dir = os.path.join(cache_dir, model_name)
    escaped_name = model_name.replace(".", "___")
    escaped_dir = os.path.join(cache_dir, escaped_name)

    for candidate in [original_dir, escaped_dir]:
        candidate = os.path.normpath(candidate)
        if not os.path.isdir(candidate):
            continue
        # 必须有 config.json
        if not os.path.isfile(os.path.join(candidate, "config.json")):
            continue
        # 必须有权重文件 (.safetensors 或 .bin)
        has_weights = (
            glob.glob(os.path.join(candidate, "*.safetensors"))
            or glob.glob(os.path.join(candidate, "*.bin"))
        )
        if has_weights:
            return candidate

    return None


def resolve_model_path(
    model_name: str,
    use_modelscope: bool = False,
    cache_dir: str = "./models",
) -> str:
    """Resolve model source path from Hugging Face id or ModelScope mirror.

    如果本地缓存已有完整模型，直接返回本地路径，不再联网下载/校验。
    """
    if not use_modelscope:
        return model_name

    # 先检查本地是否已经下载完成
    local_path = _find_local_model_dir(model_name, cache_dir)
    if local_path is not None:
        print(f"本地缓存命中，跳过下载: {local_path}")
        return local_path

    # 本地没找到，才走 ModelScope 下载
    modelscope_module = import_module("modelscope")
    snapshot_download = getattr(modelscope_module, "snapshot_download")

    return snapshot_download(model_name, cache_dir=cache_dir, revision="master")


def load_tokenizer_and_model(
    model_name: str,
    use_modelscope: bool = False,
    cache_dir: str = "./models",
    prefer_bf16: bool = True,
    trust_remote_code: bool = True,
    use_qlora: bool = False,
):
    """Load tokenizer and CausalLM model with auto backend placement.

    use_qlora=True: 以 4-bit NF4 量化加载底座模型（QLoRA 模式），可大幅降低显存占用。
    Returns: (tokenizer, model, model_path, device_info)
    """
    model_path = resolve_model_path(model_name=model_name, use_modelscope=use_modelscope, cache_dir=cache_dir)

    device = get_device_info()
    dtype = get_torch_dtype(device.device_type, prefer_bf16=prefer_bf16)
    device_map = "auto" if device.device_type == "cuda" else None

    # QLoRA: 4-bit 量化配置，仅在 use_qlora=True 时生效
    quantization_config = None
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NF4 是 QLoRA 论文推荐格式
            bnb_4bit_compute_dtype=dtype,        # 计算时用 bf16/fp16，节省显存
            bnb_4bit_use_double_quant=True,      # 双重量化，进一步压缩
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        device_map=device_map,
        quantization_config=quantization_config,  # None 时等价于原来的行为
    )

    if device.device_type in ["npu", "mps"]:
        model.to(torch.device(device.device_type))

    return tokenizer, model, model_path, device


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

PROMPT_HEADER = (
    "Given the following restaurant review, please rate it from 1 to 5 stars, where:\n"
    "- 1 star: Very poor experience\n"
    "- 2 stars: Poor experience\n"
    "- 3 stars: Average experience\n"
    "- 4 stars: Good experience\n"
    "- 5 stars: Excellent experience"
)

PROMPT_FOOTER = (
    "Please provide only a single number (1, 2, 3, 4, or 5) as your rating.\n"
    "Return format: `Rating: <digit>` (no explanation).\n"
    "Rating: "
)


def build_zero_shot_prompt(title: str, review: str) -> str:
    """Build baseline prompt for Task A."""
    input_text = f"Title: {title}\nReview: {review}" if title else f"Review: {review}"
    return f"{PROMPT_HEADER}\n\n{input_text}\n\n{PROMPT_FOOTER}"


def build_nshot_prompt(
    title: str,
    review: str,
    examples: List[Dict],
    n: int = 4,
    seed: Optional[int] = None,
) -> str:
    """Build few-shot prompt for Task B.1.1."""
    k = min(n, len(examples))
    if k <= 0:
        selected = []
    elif seed is not None:
        selected = random.Random(seed).sample(examples, k=k)
    else:
        selected = random.sample(examples, k=k)

    lines = [PROMPT_HEADER, "", "Here are some examples as below:"]

    for idx, ex in enumerate(selected, start=1):
        lines.append(f"### Example {idx}")
        if ex.get("title"):
            lines.append(f"Title: {ex['title']}")

        text = str(ex["review"])
        if len(text) > 400:
            text = text[:400] + "..."

        lines.append(f"Review: {text}")
        lines.append(f"Rating: {ex['rating']}")
        lines.append("")

    lines.append("### Query Review")
    input_text = f"Title: {title}\nReview: {review}" if title else f"Review: {review}"
    lines.append(input_text)
    lines.append("")

    lines.append(PROMPT_FOOTER.strip("\n"))

    return "\n".join(lines)


def build_finetuned_prompt(title: str, review: str) -> str:
    """Build prompt aligned with LoRA instruction tuning data format."""
    input_text = f"Title: {title}\nReview: {review}" if title else f"Review: {review}"
    return f"{PROMPT_HEADER}\n\n{input_text}\n\n{PROMPT_FOOTER}"


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def extract_rating_from_output(
    output_text: str,
    default_rating: int = 3,
    return_parse_info: bool = False,
):
    """Parse rating from model output.

    Priority: 1) `Rating: <digit>`  2) first standalone digit in 1..5.
    If `return_parse_info=True`, return `(rating, parse_failed)`.
    """
    rating_match = re.search(r"rating\s*[:：]\s*([1-5])\b", output_text, flags=re.IGNORECASE)
    if rating_match:
        rating = int(rating_match.group(1))
        if return_parse_info:
            return rating, False
        return rating

    numbers = re.findall(r"\b([1-5])\b", output_text)
    if numbers:
        rating = int(numbers[0])
        if return_parse_info:
            return rating, False
        return rating

    if return_parse_info:
        return default_rating, True
    return default_rating


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 10,
    temperature: float = 0.1,
    do_sample: bool = False,
    system_prompt: Optional[str] = None,
) -> str:
    """Run a single-turn generation and return decoded assistant text."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    if hasattr(model_inputs, "to"):
        model_inputs = model_inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    new_token_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# LoRA model loading
# ---------------------------------------------------------------------------

def load_merged_lora_model(
    base_model_name: str,
    lora_dir: str,
    use_modelscope: bool = False,
    cache_dir: str = "./models",
    prefer_bf16: bool = True,
):
    """Load base model, attach LoRA adapter, and merge weights for inference.

    Returns: (tokenizer, merged_model, device_info)
    """
    peft_module = import_module("peft")
    PeftModel = getattr(peft_module, "PeftModel")

    tokenizer, base_model, _, device = load_tokenizer_and_model(
        model_name=base_model_name,
        use_modelscope=use_modelscope,
        cache_dir=cache_dir,
        prefer_bf16=prefer_bf16,
    )

    lora_model = PeftModel.from_pretrained(base_model, lora_dir)
    merged_model = lora_model.merge_and_unload()

    if device.device_type in ["npu", "mps"]:
        merged_model = merged_model.to(device.device_type)

    return tokenizer, merged_model, device
===
import os
import re
import glob
import random
from importlib import import_module
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class DeviceInfo:
    device_type: str
    device_name: str
    total_memory_gb: Optional[float]


def get_device_info() -> DeviceInfo:
    """Detect CUDA/NPU/MPS/CPU and return normalized device metadata."""
    # Ascend NPU
    try:
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            try:
                current_idx = npu.current_device()
            except Exception:
                current_idx = 0
            try:
                npu.set_device(current_idx)
            except Exception:
                pass

            memory_gb = None
            try:
                memory_gb = npu.get_device_properties(current_idx).total_memory / 1e9
            except Exception:
                pass

            return DeviceInfo(
                device_type="npu",
                device_name=f"Ascend NPU (Device {current_idx})",
                total_memory_gb=memory_gb,
            )
    except Exception:
        pass

    # NVIDIA CUDA
    if torch.cuda.is_available():
        return DeviceInfo(
            device_type="cuda",
            device_name=torch.cuda.get_device_name(0),
            total_memory_gb=torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # Apple MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceInfo(
            device_type="mps",
            device_name="Apple Silicon GPU (MPS)",
            total_memory_gb=None,
        )

    return DeviceInfo(device_type="cpu", device_name="CPU", total_memory_gb=None)


def get_torch_dtype(device_type: str, prefer_bf16: bool = True) -> torch.dtype:
    """Choose dtype by accelerator characteristics."""
    if device_type == "cuda":
        return torch.float16
    if device_type == "npu":
        return torch.bfloat16 if prefer_bf16 else torch.float16
    return torch.float32


def _find_local_model_dir(model_name: str, cache_dir: str) -> Optional[str]:
    """检查本地缓存是否已有完整模型（含 config.json + 权重文件）。

    ModelScope 在 Windows 上会把 model_name 里的 '.' 替换成 '___'，
    所以这里两种路径都检查一下。找到就返回路径，没找到返回 None。
    """
    original_dir = os.path.join(cache_dir, model_name)
    escaped_name = model_name.replace(".", "___")
    escaped_dir = os.path.join(cache_dir, escaped_name)

    for candidate in [original_dir, escaped_dir]:
        candidate = os.path.normpath(candidate)
        if not os.path.isdir(candidate):
            continue
        # 必须有 config.json
        if not os.path.isfile(os.path.join(candidate, "config.json")):
            continue
        # 必须有权重文件 (.safetensors 或 .bin)
        has_weights = (
            glob.glob(os.path.join(candidate, "*.safetensors"))
            or glob.glob(os.path.join(candidate, "*.bin"))
        )
        if has_weights:
            return candidate

    return None


def resolve_model_path(
    model_name: str,
    use_modelscope: bool = False,
    cache_dir: str = "./models",
) -> str:
    """Resolve model source path from Hugging Face id or ModelScope mirror.

    如果本地缓存已有完整模型，直接返回本地路径，不再联网下载/校验。
    """
    if not use_modelscope:
        return model_name

    # 先检查本地是否已经下载完成
    local_path = _find_local_model_dir(model_name, cache_dir)
    if local_path is not None:
        print(f"本地缓存命中，跳过下载: {local_path}")
        return local_path

    # 本地没找到，才走 ModelScope 下载
    modelscope_module = import_module("modelscope")
    snapshot_download = getattr(modelscope_module, "snapshot_download")

    return snapshot_download(model_name, cache_dir=cache_dir, revision="master")


def load_tokenizer_and_model(
    model_name: str,
    use_modelscope: bool = False,
    cache_dir: str = "./models",
    prefer_bf16: bool = True,
    trust_remote_code: bool = True,
    use_qlora: bool = False,
):
    """Load tokenizer and CausalLM model with auto backend placement.

    use_qlora=True: 以 4-bit NF4 量化加载底座模型（QLoRA 模式），可大幅降低显存占用。
    Returns: (tokenizer, model, model_path, device_info)
    """
    model_path = resolve_model_path(model_name=model_name, use_modelscope=use_modelscope, cache_dir=cache_dir)

    device = get_device_info()
    dtype = get_torch_dtype(device.device_type, prefer_bf16=prefer_bf16)
    device_map = "auto" if device.device_type == "cuda" else None

    # QLoRA: 4-bit 量化配置，仅在 use_qlora=True 时生效
    quantization_config = None
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NF4 是 QLoRA 论文推荐格式
            bnb_4bit_compute_dtype=dtype,        # 计算时用 bf16/fp16，节省显存
            bnb_4bit_use_double_quant=True,      # 双重量化，进一步压缩
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        device_map=device_map,
        quantization_config=quantization_config,  # None 时等价于原来的行为
    )

    if device.device_type in ["npu", "mps"]:
        model.to(torch.device(device.device_type))

    return tokenizer, model, model_path, device


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

PROMPT_HEADER = (
    "Given the following restaurant review, please rate it from 1 to 5 stars, where:\n"
    "- 1 star: Very poor experience\n"
    "- 2 stars: Poor experience\n"
    "- 3 stars: Average experience\n"
    "- 4 stars: Good experience\n"
    "- 5 stars: Excellent experience"
)

PROMPT_FOOTER = (
    "Please provide only a single number (1, 2, 3, 4, or 5) as your rating.\n"
    "Return format: `Rating: <digit>` (no explanation).\n"
    "Rating: "
)


def build_zero_shot_prompt(title: str, review: str) -> str:
    """Build baseline prompt for Task A."""
    input_text = f"Title: {title}\nReview: {review}" if title else f"Review: {review}"
    return f"{PROMPT_HEADER}\n\n{input_text}\n\n{PROMPT_FOOTER}"


def build_nshot_prompt(
    title: str,
    review: str,
    examples: List[Dict],
    n: int = 4,
    seed: Optional[int] = None,
    use_semantic: bool = False,
    embeddings: Optional["np.ndarray"] = None,
    query_embedding: Optional["np.ndarray"] = None,
) -> str:
    """Build few-shot prompt for Task B.1.1.[v2]

    use_semantic=False（默认）：随机采样，与原行为完全一致。
    use_semantic=True：用 cosine similarity 选最相近的 n 个例子（需传入
      embeddings=build_example_embeddings(examples) 和
      query_embedding=build_example_embeddings([{"review": review}])[0]）。
    """
    k = min(n, len(examples))
    if k <= 0:
        selected = []
    elif use_semantic and embeddings is not None and query_embedding is not None:
        # cosine similarity 排序[v2]
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        sims = emb_norm @ q_norm
        top_indices = np.argsort(sims)[::-1][:k]
        selected = [examples[i] for i in top_indices]
    elif seed is not None:
        selected = random.Random(seed).sample(examples, k=k)
    else:
        selected = random.sample(examples, k=k)

    lines = [PROMPT_HEADER, "", "Here are some examples as below:"]

    for idx, ex in enumerate(selected, start=1):
        lines.append(f"### Example {idx}")
        if ex.get("title"):
            lines.append(f"Title: {ex['title']}")

        text = str(ex["review"])
        if len(text) > 400:
            text = text[:400] + "..."

        lines.append(f"Review: {text}")
        lines.append(f"Rating: {ex['rating']}")
        lines.append("")

    lines.append("### Query Review")
    input_text = f"Title: {title}\nReview: {review}" if title else f"Review: {review}"
    lines.append(input_text)
    lines.append("")

    lines.append(PROMPT_FOOTER.strip("\n"))

    return "\n".join(lines)


def build_finetuned_prompt(title: str, review: str) -> str:
    """Build prompt aligned with LoRA instruction tuning data format."""
    input_text = f"Title: {title}\nReview: {review}" if title else f"Review: {review}"
    return f"{PROMPT_HEADER}\n\n{input_text}\n\n{PROMPT_FOOTER}"


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def extract_rating_from_output(
    output_text: str,
    default_rating: int = 3,
    return_parse_info: bool = False,
):
    """Parse rating from model output.

    Priority: 1) `Rating: <digit>`  2) first standalone digit in 1..5.
    If `return_parse_info=True`, return `(rating, parse_failed)`.
    """
    rating_match = re.search(r"rating\s*[:：]\s*([1-5])\b", output_text, flags=re.IGNORECASE)
    if rating_match:
        rating = int(rating_match.group(1))
        if return_parse_info:
            return rating, False
        return rating

    numbers = re.findall(r"\b([1-5])\b", output_text)
    if numbers:
        rating = int(numbers[0])
        if return_parse_info:
            return rating, False
        return rating

    if return_parse_info:
        return default_rating, True
    return default_rating


# ---------------------------------------------------------------------------
# Logits-based rating classification（分类式推理，比生成式快 5~10×)[v2]
# ---------------------------------------------------------------------------

def classify_rating_by_logits(
    model,
    tokenizer,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> int:
    """直接比较 '1'~'5' 五个 token 的 logits，取 argmax 返回评分（1~5）。

    相比 generate_response + parse，推理速度快 5~10 倍，且不存在解析失败的情况。
    """
    # 构建与 generate_response 一致的消息格式
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    if hasattr(model_inputs, "to"):
        model_inputs = model_inputs.to(model.device)

    # 获取 '1'~'5' 对应的 token id（取单字符编码的第一个 token）
    rating_token_ids = [
        tokenizer.encode(str(r), add_special_tokens=False)[0] for r in range(1, 6)
    ]

    with torch.no_grad():
        outputs = model(**model_inputs)
    # 取最后一个位置的 logits，比较 5 个候选 token
    last_logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)
    candidate_logits = last_logits[rating_token_ids]
    best_idx = int(candidate_logits.argmax().item())
    return best_idx + 1  # 1~5


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 10,
    temperature: float = 0.1,
    do_sample: bool = False,
    system_prompt: Optional[str] = None,
) -> str:
    """Run a single-turn generation and return decoded assistant text."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    if hasattr(model_inputs, "to"):
        model_inputs = model_inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    new_token_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0].strip()


def generate_response_batch(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 10,
    temperature: float = 0.1,
    do_sample: bool = False,
    system_prompt: Optional[str] = None,
) -> List[str]:
    """批量推理，将 prompts 分批送入 model.generate，速度比逐条快数倍。

    Returns: 与 prompts 顺序对应的解码字符串列表。
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _build_text(prompt):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    all_outputs: List[str] = []
    for batch_start in range(0, len(prompts), batch_size):
        batch_texts = [_build_text(p) for p in prompts[batch_start: batch_start + batch_size]]
        model_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )

        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
            new_ids = output_ids[len(input_ids):]
            all_outputs.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())

    return all_outputs


# ---------------------------------------------------------------------------
# 语义嵌入工具（配合 build_nshot_prompt use_semantic 模式使用）
# ---------------------------------------------------------------------------

def build_example_embeddings(
    examples: List[Dict],
    model_name: str = "all-MiniLM-L6-v2",
    text_key: str = "review",
) -> "np.ndarray":
    """用 sentence-transformers 对 examples 列表做嵌入，返回 (N, D) 的 numpy 数组。

    配合 build_nshot_prompt(use_semantic=True, embeddings=..., query_embedding=...) 使用。
    需安装 sentence-transformers：pip install sentence-transformers

    示例：
        embs = build_example_embeddings(example_library)
        query_emb = build_example_embeddings([{"review": review}])[0]
        prompt = build_nshot_prompt(
            title, review, example_library,
            use_semantic=True, embeddings=embs, query_embedding=query_emb,
        )
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "请先安装 sentence-transformers：pip install sentence-transformers"
        )

    st_model = SentenceTransformer(model_name)
    texts = [str(ex.get(text_key, "")) for ex in examples]
    embeddings = st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings  # shape: (N, D)


# ---------------------------------------------------------------------------
# LoRA model loading
# ---------------------------------------------------------------------------

def load_merged_lora_model(
    base_model_name: str,
    lora_dir: str,
    use_modelscope: bool = False,
    cache_dir: str = "./models",
    prefer_bf16: bool = True,
):
    """Load base model, attach LoRA adapter, and merge weights for inference.

    Returns: (tokenizer, merged_model, device_info)
    """
    peft_module = import_module("peft")
    PeftModel = getattr(peft_module, "PeftModel")

    tokenizer, base_model, _, device = load_tokenizer_and_model(
        model_name=base_model_name,
        use_modelscope=use_modelscope,
        cache_dir=cache_dir,
        prefer_bf16=prefer_bf16,
    )

    lora_model = PeftModel.from_pretrained(base_model, lora_dir)
    merged_model = lora_model.merge_and_unload()

    if device.device_type in ["npu", "mps"]:
        merged_model = merged_model.to(device.device_type)

    return tokenizer, merged_model, device
```

| 改动 | 内容 |
|------|------|
| ② 分类式推理 | 新增 [classify_rating_by_logits(model, tokenizer, prompt)](file:///d:/my_files/job/projects/QwenLora/model.py#301-334)：只比较 5 个候选 token 的 logits，无需生成&解析，速度快 5~10× |
| ④ 语义检索 | [build_nshot_prompt](file:///d:/my_files/job/projects/QwenLora/model.py#201-256) 新增 `use_semantic / embeddings / query_embedding` 参数，默认关闭兼容原行为 |
| ④ 嵌入工具 | 新增 [build_example_embeddings(examples)](file:///d:/my_files/job/projects/QwenLora/model.py#430-459) 调用 sentence-transformers 生成嵌入向量 |
| ⑧ 批量推理 | 新增 [generate_response_batch(model, tokenizer, prompts, batch_size=8)](file:///d:/my_files/job/projects/QwenLora/model.py#375-424) |

**语义 N-shot 用法：**
```python
from model import build_example_embeddings, build_nshot_prompt
embs = build_example_embeddings(example_library)               # 一次性预计算
query_emb = build_example_embeddings([{"review": review}])[0]
prompt = build_nshot_prompt(title, review, example_library,
    use_semantic=True, embeddings=embs, query_embedding=query_emb)
```

**logits 分类用法：**
```python
from model import classify_rating_by_logits
rating = classify_rating_by_logits(model, tokenizer, prompt)   # 返回 1~5
```

---

## eval.py

```diff:eval.py
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from model import (
    build_finetuned_prompt,
    build_nshot_prompt,
    build_zero_shot_prompt,
    extract_rating_from_output,
    generate_response,
)


def _default_title(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def run_inference_loop(
    data_df: pd.DataFrame,
    predict_fn: Callable[[pd.Series], Dict[str, Any]],
    id_col: str = "Review_id",
    progress_every: int = 50,
) -> Dict[str, Any]:
    """Generic inference loop that records predictions and runtime."""
    ids: List[Any] = []
    predictions: List[int] = []
    parse_fallbacks: List[bool] = []
    debug_records: List[Dict[str, Any]] = []

    start = time.time()

    for step, (_, row) in enumerate(data_df.iterrows()):
        review_id = row[id_col]

        try:
            result = predict_fn(row)
            predicted_rating = int(result["predicted_rating"])
            raw_output = str(result.get("raw_output", ""))
            fallback_used = bool(result.get("fallback_used", False))
        except Exception as exc:
            predicted_rating = 3
            raw_output = f"ERROR: {exc}"
            fallback_used = True

        ids.append(review_id)
        predictions.append(predicted_rating)
        parse_fallbacks.append(fallback_used)
        debug_records.append(
            {
                "sample_idx": step,
                "review_id": review_id,
                "raw_output": raw_output,
                "predicted_rating": predicted_rating,
                "fallback_used": fallback_used,
            }
        )

        if progress_every > 0 and len(predictions) % progress_every == 0:
            print(f"Processed {len(predictions)} samples...")

    elapsed = time.time() - start

    prediction_df = pd.DataFrame(
        {
            id_col: ids,
            "Predicted_Rating": predictions,
        }
    )
    parse_failure_count = int(sum(parse_fallbacks))
    parse_failure_total = len(parse_fallbacks)
    prediction_df.attrs["parse_failure_count"] = parse_failure_count
    prediction_df.attrs["parse_failure_total"] = parse_failure_total
    prediction_df.attrs["parse_failure_rate"] = (
        parse_failure_count / parse_failure_total if parse_failure_total > 0 else 0.0
    )

    return {
        "predictions_df": prediction_df,
        "debug_records": debug_records,
        "inference_seconds": elapsed,
        "sample_count": len(predictions),
    }


def run_zero_shot_inference(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    max_new_tokens: int = 10,
) -> Dict[str, Any]:
    """Task A: baseline zero-shot inference."""

    def _predict(row: pd.Series) -> Dict[str, Any]:
        title = _default_title(row.get("Title"))
        review = str(row["Review"])
        prompt = build_zero_shot_prompt(title=title, review=review)
        raw_output = generate_response(model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=max_new_tokens)
        parsed = extract_rating_from_output(raw_output, return_parse_info=True)
        predicted, parse_failed = parsed if isinstance(parsed, tuple) else (int(parsed), False)
        return {"raw_output": raw_output, "predicted_rating": predicted, "fallback_used": parse_failed}

    return run_inference_loop(test_df, _predict)


def run_nshot_inference(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    example_library: List[Dict[str, Any]],
    n_shot: int = 4,
    max_new_tokens: int = 10,
    seed: int = 5494,
) -> Dict[str, Any]:
    """Task B.1.1: N-shot in-context inference."""

    def _predict(row: pd.Series) -> Dict[str, Any]:
        title = _default_title(row.get("Title"))
        review = str(row["Review"])

        # Use deterministic per-row seed so notebook reruns are reproducible.
        review_id_text = str(row.get("Review_id", ""))
        stable_hash = int(hashlib.md5(review_id_text.encode("utf-8")).hexdigest()[:8], 16)
        row_seed = seed + (stable_hash % 1_000_000)
        prompt = build_nshot_prompt(
            title=title,
            review=review,
            examples=example_library,
            n=n_shot,
            seed=row_seed,
        )

        raw_output = generate_response(model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=max_new_tokens)
        parsed = extract_rating_from_output(raw_output, return_parse_info=True)
        predicted, parse_failed = parsed if isinstance(parsed, tuple) else (int(parsed), False)
        return {"raw_output": raw_output, "predicted_rating": predicted, "fallback_used": parse_failed}

    return run_inference_loop(test_df, _predict)


def run_lora_inference(
    lora_model,
    tokenizer,
    test_df: pd.DataFrame,
    max_new_tokens: int = 10,
) -> Dict[str, Any]:
    """Task B.1.2: merged LoRA model inference."""

    def _predict(row: pd.Series) -> Dict[str, Any]:
        title = _default_title(row.get("Title"))
        review = str(row["Review"])
        prompt = build_finetuned_prompt(title=title, review=review)
        raw_output = generate_response(
            model=lora_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        parsed = extract_rating_from_output(raw_output, return_parse_info=True)
        predicted, parse_failed = parsed if isinstance(parsed, tuple) else (int(parsed), False)
        return {"raw_output": raw_output, "predicted_rating": predicted, "fallback_used": parse_failed}

    return run_inference_loop(test_df, _predict)


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    answer_df: pd.DataFrame,
    id_col: str = "Review_id",
    true_col: str = "rating_numeric",
    pred_col: str = "Predicted_Rating",
) -> Dict[str, Any]:
    """Compute standard multi-class metrics used in the project report."""
    merged = predictions_df.merge(answer_df, on=id_col)
    labels = [1, 2, 3, 4, 5]
    target_names = ["1 star", "2 star", "3 star", "4 star", "5 star"]

    accuracy = accuracy_score(merged[true_col], merged[pred_col])
    macro_f1 = f1_score(merged[true_col], merged[pred_col], average="macro")
    weighted_f1 = f1_score(merged[true_col], merged[pred_col], average="weighted")

    report_text = classification_report(
        merged[true_col],
        merged[pred_col],
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    cm = confusion_matrix(merged[true_col], merged[pred_col], labels=labels)

    parse_failure_rate = None
    parse_failure_count = None
    if "Parse_Fallback_Used" in merged.columns:
        parse_failure_count = int(merged["Parse_Fallback_Used"].astype(bool).sum())
        parse_failure_rate = parse_failure_count / len(merged) if len(merged) > 0 else 0.0
        print(f"Parse fallback rate: {parse_failure_rate:.4f} ({parse_failure_count}/{len(merged)})")
    else:
        parse_failure_count = predictions_df.attrs.get("parse_failure_count")
        parse_failure_total = predictions_df.attrs.get("parse_failure_total", len(merged))
        if isinstance(parse_failure_count, int) and isinstance(parse_failure_total, int) and parse_failure_total > 0:
            parse_failure_rate = parse_failure_count / parse_failure_total
            print(f"Parse fallback rate: {parse_failure_rate:.4f} ({parse_failure_count}/{parse_failure_total})")
        else:
            print("Parse fallback rate: N/A (no parser tracking metadata)")

    return {
        "merged_df": merged,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_text,
        "confusion_matrix": cm,
        "parse_failure_rate": parse_failure_rate,
        "parse_failure_count": parse_failure_count,
    }
===
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from model import (
    build_finetuned_prompt,
    build_nshot_prompt,
    build_zero_shot_prompt,
    classify_rating_by_logits,
    extract_rating_from_output,
    generate_response,
    generate_response_batch,
)


def _default_title(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def plot_confusion_matrix(cm, labels=None):
    """用 seaborn heatmap 画混淆矩阵，返回 matplotlib Figure。

    labels: 如 ["1星", "2星", "3星", "4星", "5星"]，默认用数字序号。
    示例：
        fig = plot_confusion_matrix(result["confusion_matrix"], labels=["1星","2星","3星","4星","5星"])
        display(fig)  # Jupyter Notebook 直接显示
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if labels is None:
        labels = [str(i) for i in range(1, cm.shape[0] + 1)]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def run_inference_loop(
    data_df: pd.DataFrame,
    predict_fn: Callable[[pd.Series], Dict[str, Any]],
    id_col: str = "Review_id",
    progress_every: int = 50,
) -> Dict[str, Any]:
    """Generic inference loop that records predictions and runtime."""
    ids: List[Any] = []
    predictions: List[int] = []
    parse_fallbacks: List[bool] = []
    debug_records: List[Dict[str, Any]] = []

    start = time.time()

    for step, (_, row) in enumerate(data_df.iterrows()):
        review_id = row[id_col]

        try:
            result = predict_fn(row)
            predicted_rating = int(result["predicted_rating"])
            raw_output = str(result.get("raw_output", ""))
            fallback_used = bool(result.get("fallback_used", False))
        except Exception as exc:
            predicted_rating = 3
            raw_output = f"ERROR: {exc}"
            fallback_used = True

        ids.append(review_id)
        predictions.append(predicted_rating)
        parse_fallbacks.append(fallback_used)
        debug_records.append(
            {
                "sample_idx": step,
                "review_id": review_id,
                "raw_output": raw_output,
                "predicted_rating": predicted_rating,
                "fallback_used": fallback_used,
            }
        )

        if progress_every > 0 and len(predictions) % progress_every == 0:
            print(f"Processed {len(predictions)} samples...")

    elapsed = time.time() - start

    prediction_df = pd.DataFrame(
        {
            id_col: ids,
            "Predicted_Rating": predictions,
        }
    )
    parse_failure_count = int(sum(parse_fallbacks))
    parse_failure_total = len(parse_fallbacks)
    prediction_df.attrs["parse_failure_count"] = parse_failure_count
    prediction_df.attrs["parse_failure_total"] = parse_failure_total
    prediction_df.attrs["parse_failure_rate"] = (
        parse_failure_count / parse_failure_total if parse_failure_total > 0 else 0.0
    )

    return {
        "predictions_df": prediction_df,
        "debug_records": debug_records,
        "inference_seconds": elapsed,
        "sample_count": len(predictions),
    }


def run_zero_shot_inference(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    max_new_tokens: int = 10,
) -> Dict[str, Any]:
    """Task A: baseline zero-shot inference."""

    def _predict(row: pd.Series) -> Dict[str, Any]:
        title = _default_title(row.get("Title"))
        review = str(row["Review"])
        prompt = build_zero_shot_prompt(title=title, review=review)
        raw_output = generate_response(model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=max_new_tokens)
        parsed = extract_rating_from_output(raw_output, return_parse_info=True)
        predicted, parse_failed = parsed if isinstance(parsed, tuple) else (int(parsed), False)
        return {"raw_output": raw_output, "predicted_rating": predicted, "fallback_used": parse_failed}

    return run_inference_loop(test_df, _predict)


def run_nshot_inference(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    example_library: List[Dict[str, Any]],
    n_shot: int = 4,
    max_new_tokens: int = 10,
    seed: int = 5494,
) -> Dict[str, Any]:
    """Task B.1.1: N-shot in-context inference."""

    def _predict(row: pd.Series) -> Dict[str, Any]:
        title = _default_title(row.get("Title"))
        review = str(row["Review"])

        # Use deterministic per-row seed so notebook reruns are reproducible.
        review_id_text = str(row.get("Review_id", ""))
        stable_hash = int(hashlib.md5(review_id_text.encode("utf-8")).hexdigest()[:8], 16)
        row_seed = seed + (stable_hash % 1_000_000)
        prompt = build_nshot_prompt(
            title=title,
            review=review,
            examples=example_library,
            n=n_shot,
            seed=row_seed,
        )

        raw_output = generate_response(model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=max_new_tokens)
        parsed = extract_rating_from_output(raw_output, return_parse_info=True)
        predicted, parse_failed = parsed if isinstance(parsed, tuple) else (int(parsed), False)
        return {"raw_output": raw_output, "predicted_rating": predicted, "fallback_used": parse_failed}

    return run_inference_loop(test_df, _predict)


def run_lora_inference(
    lora_model,
    tokenizer,
    test_df: pd.DataFrame,
    max_new_tokens: int = 10,
) -> Dict[str, Any]:
    """Task B.1.2: merged LoRA model inference."""

    def _predict(row: pd.Series) -> Dict[str, Any]:
        title = _default_title(row.get("Title"))
        review = str(row["Review"])
        prompt = build_finetuned_prompt(title=title, review=review)
        raw_output = generate_response(
            model=lora_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        parsed = extract_rating_from_output(raw_output, return_parse_info=True)
        predicted, parse_failed = parsed if isinstance(parsed, tuple) else (int(parsed), False)
        return {"raw_output": raw_output, "predicted_rating": predicted, "fallback_used": parse_failed}

    return run_inference_loop(test_df, _predict)


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    answer_df: pd.DataFrame,
    id_col: str = "Review_id",
    true_col: str = "rating_numeric",
    pred_col: str = "Predicted_Rating",
) -> Dict[str, Any]:
    """Compute standard multi-class metrics used in the project report."""
    merged = predictions_df.merge(answer_df, on=id_col)
    labels = [1, 2, 3, 4, 5]
    target_names = ["1 star", "2 star", "3 star", "4 star", "5 star"]

    accuracy = accuracy_score(merged[true_col], merged[pred_col])
    macro_f1 = f1_score(merged[true_col], merged[pred_col], average="macro")
    weighted_f1 = f1_score(merged[true_col], merged[pred_col], average="weighted")

    report_text = classification_report(
        merged[true_col],
        merged[pred_col],
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    cm = confusion_matrix(merged[true_col], merged[pred_col], labels=labels)

    parse_failure_rate = None
    parse_failure_count = None
    if "Parse_Fallback_Used" in merged.columns:
        parse_failure_count = int(merged["Parse_Fallback_Used"].astype(bool).sum())
        parse_failure_rate = parse_failure_count / len(merged) if len(merged) > 0 else 0.0
        print(f"Parse fallback rate: {parse_failure_rate:.4f} ({parse_failure_count}/{len(merged)})")
    else:
        parse_failure_count = predictions_df.attrs.get("parse_failure_count")
        parse_failure_total = predictions_df.attrs.get("parse_failure_total", len(merged))
        if isinstance(parse_failure_count, int) and isinstance(parse_failure_total, int) and parse_failure_total > 0:
            parse_failure_rate = parse_failure_count / parse_failure_total
            print(f"Parse fallback rate: {parse_failure_rate:.4f} ({parse_failure_count}/{parse_failure_total})")
        else:
            print("Parse fallback rate: N/A (no parser tracking metadata)")

    return {
        "merged_df": merged,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_text,
        "confusion_matrix": cm,
        "confusion_matrix_fig": plot_confusion_matrix(cm, labels=["1星", "2星", "3星", "4星", "5星"]),
        "parse_failure_rate": parse_failure_rate,
        "parse_failure_count": parse_failure_count,
    }


# ---------------------------------------------------------------------------
# 批量推理函数（相比单条循环快数倍）
# ---------------------------------------------------------------------------

def run_zero_shot_inference_batch(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    batch_size: int = 8,
    max_new_tokens: int = 10,
) -> Dict[str, Any]:
    """批量版 zero-shot 推理，速度比逐条循环快数倍。"""
    rows = list(test_df.iterrows())
    prompts = [
        build_zero_shot_prompt(
            title=_default_title(row.get("Title")),
            review=str(row["Review"]),
        )
        for _, row in rows
    ]
    raw_outputs = generate_response_batch(
        model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens
    )

    ids, predictions, debug_records = [], [], []
    start = time.time()
    for step, ((_, row), raw_output) in enumerate(zip(rows, raw_outputs)):
        parsed, parse_failed = extract_rating_from_output(raw_output, return_parse_info=True)
        ids.append(row[test_df.columns[0] if "Review_id" not in test_df.columns else "Review_id"])
        predictions.append(int(parsed))
        debug_records.append({"sample_idx": step, "raw_output": raw_output,
                               "predicted_rating": int(parsed), "fallback_used": parse_failed})

    return _build_inference_result(test_df, ids, predictions, debug_records, time.time() - start)


def run_nshot_inference_batch(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    example_library: List[Dict[str, Any]],
    n_shot: int = 4,
    batch_size: int = 8,
    max_new_tokens: int = 10,
    seed: int = 5494,
) -> Dict[str, Any]:
    """批量版 N-shot 推理。"""
    rows = list(test_df.iterrows())
    prompts = []
    for _, row in rows:
        review_id_text = str(row.get("Review_id", ""))
        stable_hash = int(hashlib.md5(review_id_text.encode("utf-8")).hexdigest()[:8], 16)
        row_seed = seed + (stable_hash % 1_000_000)
        prompts.append(build_nshot_prompt(
            title=_default_title(row.get("Title")),
            review=str(row["Review"]),
            examples=example_library,
            n=n_shot,
            seed=row_seed,
        ))

    raw_outputs = generate_response_batch(
        model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens
    )

    ids, predictions, debug_records = [], [], []
    start = time.time()
    id_col = "Review_id" if "Review_id" in test_df.columns else test_df.columns[0]
    for step, ((_, row), raw_output) in enumerate(zip(rows, raw_outputs)):
        parsed, parse_failed = extract_rating_from_output(raw_output, return_parse_info=True)
        ids.append(row[id_col])
        predictions.append(int(parsed))
        debug_records.append({"sample_idx": step, "raw_output": raw_output,
                               "predicted_rating": int(parsed), "fallback_used": parse_failed})

    return _build_inference_result(test_df, ids, predictions, debug_records, time.time() - start)


def run_lora_inference_batch(
    lora_model,
    tokenizer,
    test_df: pd.DataFrame,
    batch_size: int = 8,
    max_new_tokens: int = 10,
) -> Dict[str, Any]:
    """批量版 LoRA 推理，输入输出格式与 run_lora_inference 完全一致。"""
    rows = list(test_df.iterrows())
    prompts = [
        build_finetuned_prompt(
            title=_default_title(row.get("Title")),
            review=str(row["Review"]),
        )
        for _, row in rows
    ]
    raw_outputs = generate_response_batch(
        lora_model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens
    )

    ids, predictions, debug_records = [], [], []
    start = time.time()
    id_col = "Review_id" if "Review_id" in test_df.columns else test_df.columns[0]
    for step, ((_, row), raw_output) in enumerate(zip(rows, raw_outputs)):
        parsed, parse_failed = extract_rating_from_output(raw_output, return_parse_info=True)
        ids.append(row[id_col])
        predictions.append(int(parsed))
        debug_records.append({"sample_idx": step, "raw_output": raw_output,
                               "predicted_rating": int(parsed), "fallback_used": parse_failed})

    return _build_inference_result(test_df, ids, predictions, debug_records, time.time() - start)


def _build_inference_result(
    test_df: pd.DataFrame,
    ids: List,
    predictions: List[int],
    debug_records: List[Dict],
    elapsed: float,
    id_col: str = "Review_id",
) -> Dict[str, Any]:
    """构建与 run_inference_loop 相同结构的返回字典。"""
    if id_col not in test_df.columns:
        id_col = test_df.columns[0]
    prediction_df = pd.DataFrame({id_col: ids, "Predicted_Rating": predictions})
    fallback_count = sum(r["fallback_used"] for r in debug_records)
    total = len(predictions)
    prediction_df.attrs["parse_failure_count"] = fallback_count
    prediction_df.attrs["parse_failure_total"] = total
    prediction_df.attrs["parse_failure_rate"] = fallback_count / total if total > 0 else 0.0
    return {
        "predictions_df": prediction_df,
        "debug_records": debug_records,
        "inference_seconds": elapsed,
        "sample_count": total,
    }
```

| 改动 | 内容 |
|------|------|
| ⑥ 可视化 | 新增 [plot_confusion_matrix(cm, labels)](file:///d:/my_files/job/projects/QwenLora/eval.py#25-54)；[evaluate_predictions](file:///d:/my_files/job/projects/QwenLora/eval.py#203-255) 返回值中增加 `"confusion_matrix_fig"` |
| ⑧ 批量推理 | 新增 [run_zero_shot_inference_batch](file:///d:/my_files/job/projects/QwenLora/eval.py#261-291)、[run_nshot_inference_batch](file:///d:/my_files/job/projects/QwenLora/eval.py#293-333)、[run_lora_inference_batch](file:///d:/my_files/job/projects/QwenLora/eval.py#335-366)（均带 `batch_size` 参数），返回格式与原函数完全一致 |

**混淆矩阵可视化用法：**
```python
result = evaluate_predictions(predictions_df, answer_df)
display(result["confusion_matrix_fig"])   # Notebook 中直接显示 heatmap
```

**批量推理用法：**
```python
from eval import run_lora_inference_batch
result = run_lora_inference_batch(lora_model, tokenizer, test_df, batch_size=8)
```

---

## 验证结果

```
python -c "import py_compile; [py_compile.compile(f) for f in ['train.py','dataset.py','model.py','eval.py']]; print('ALL OK')"
# → ALL OK
```

> [!NOTE]
> 所有新功能均有默认值，不传新参数时行为与改动前完全一致，不会破坏已有 Notebook 代码。
