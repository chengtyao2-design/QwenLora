import hashlib
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from model import (
    build_finetuned_prompt,
    build_nshot_prompt,
    build_zero_shot_prompt,
    classify_rating_by_logits, #[v2]
    extract_rating_from_output,
    generate_response,
    generate_response_batch, #[v2]
)


def _default_title(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def plot_confusion_matrix(cm, labels=None):
    """用 seaborn heatmap 画混淆矩阵，返回 matplotlib Figure。[v2]

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
        "confusion_matrix_fig": plot_confusion_matrix(cm, labels=["1星", "2星", "3星", "4星", "5星"]), #[v2]
        "parse_failure_rate": parse_failure_rate,
        "parse_failure_count": parse_failure_count,
    }


# ---------------------------------------------------------------------------
# 批量推理函数（相比单条循环快数倍）[v2]
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
