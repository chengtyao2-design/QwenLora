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
        predicted = extract_rating_from_output(raw_output)
        return {"raw_output": raw_output, "predicted_rating": predicted, "fallback_used": False}

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
        predicted = extract_rating_from_output(raw_output)
        return {"raw_output": raw_output, "predicted_rating": predicted, "fallback_used": False}

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
            system_prompt="Return only integers 1-5 with no explanation.",
        )
        predicted = extract_rating_from_output(raw_output)
        return {"raw_output": raw_output, "predicted_rating": predicted}

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

    return {
        "merged_df": merged,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_text,
        "confusion_matrix": cm,
    }
