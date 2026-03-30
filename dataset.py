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

	oversample=True: 对少数类做随机重复采样，使各类数量与最多类持平，[v2]
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

	# 按类别分组[v2]
	groups: Dict[int, List] = {r: [] for r in range(1, 6)}
	for _, row in df.iterrows():
		groups[int(row[rating_col])].append(row)

	if not oversample:
		rows = [_row_to_record(row) for rows_in_group in groups.values() for row in rows_in_group]
		return rows

	# 过采样：以最多类数量为目标，对其他类随机重复抽取[v2]
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
