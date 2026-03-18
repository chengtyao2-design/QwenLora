import os
import re
import glob
import random
from importlib import import_module
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
):
    """Load tokenizer and CausalLM model with auto backend placement.

    Returns: (tokenizer, model, model_path, device_info)
    """
    model_path = resolve_model_path(model_name=model_name, use_modelscope=use_modelscope, cache_dir=cache_dir)

    device = get_device_info()
    dtype = get_torch_dtype(device.device_type, prefer_bf16=prefer_bf16)
    device_map = "auto" if device.device_type == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map=device_map,
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
