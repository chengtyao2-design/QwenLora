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


def balanced_sample(
	df: pd.DataFrame,
	label_col: Optional[str],
	sample_size: Optional[int],
	seed: int,
	min_items_per_class: int = 1,
) -> pd.DataFrame:
	"""Sample with class coverage when labels are available."""
	if sample_size in (None, -1) or sample_size >= len(df):
		return df.reset_index(drop=True)

	if not label_col or label_col not in df.columns:
		return df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)

	label_series = df[label_col].dropna()
	if label_series.empty:
		return df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)

	labels = sorted(label_series.unique())
	mandatory_chunks = []
	remaining_df = df.copy()

	for idx, label in enumerate(labels):
		class_df = remaining_df[remaining_df[label_col] == label]
		if class_df.empty:
			continue

		take = min(min_items_per_class, len(class_df))
		chosen = class_df.sample(n=take, random_state=seed + idx)
		mandatory_chunks.append(chosen)
		remaining_df = remaining_df.drop(chosen.index)

	base_count = sum(len(chunk) for chunk in mandatory_chunks)
	target_count = max(sample_size, base_count)
	remaining_needed = max(target_count - base_count, 0)

	if remaining_needed > 0 and not remaining_df.empty:
		extra = remaining_df.sample(n=min(remaining_needed, len(remaining_df)), random_state=seed)
		mandatory_chunks.append(extra)

	if mandatory_chunks:
		sampled = pd.concat(mandatory_chunks)
	else:
		sampled = df.sample(n=min(sample_size, len(df)), random_state=seed)

	return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


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
	"""Stratified train/validation split with optional balanced downsampling."""
	train_subset, val_subset = train_test_split(
		train_df,
		test_size=test_size,
		random_state=seed,
		stratify=train_df[label_col],
	)

	if debug_sample_size not in (None, -1):
		train_subset = balanced_sample(train_subset, label_col, debug_sample_size, seed)
		val_subset = balanced_sample(val_subset, label_col, debug_sample_size, seed)

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
	rows: List[Dict[str, str]] = []

	for _, row in df.iterrows():
		title = str(row["Title"]) if pd.notna(row.get("Title")) else ""
		review = str(row["Review"])
		rating = int(row[rating_col])

		instruction = "Given the following restaurant review, rate it from 1 to 5 stars."
		input_text = f"Title: {title}\nReview: {review}" if title else f"Review: {review}"

		rows.append(
			{
				"instruction": instruction,
				"input": input_text,
				"output": str(rating),
			}
		)

	return rows


def select_instruction_subset(
	records: List[Dict[str, str]],
	max_samples: Optional[int],
) -> List[Dict[str, str]]:
	"""Take the first max_samples records, or all when disabled."""
	if max_samples in (None, -1):
		return records
	return records[: max(0, max_samples)]
