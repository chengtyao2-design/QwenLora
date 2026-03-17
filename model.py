import re
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


def get_device_map(device_type: str) -> Optional[str]:
    """Return transformers device_map strategy for each backend."""
    if device_type == "cuda":
        return "auto"
    return None


def get_torch_dtype(device_type: str, prefer_bf16: bool = True) -> torch.dtype:
    """Choose dtype by accelerator characteristics."""
    if device_type == "cuda":
        return torch.float16
    if device_type == "npu":
        return torch.bfloat16 if prefer_bf16 else torch.float16
    if device_type == "mps":
        # MPS is usually more stable with fp32.
        return torch.float32
    return torch.float32


def resolve_model_path(
    model_name: str,
    use_modelscope: bool = False,
    cache_dir: str = "./models",
) -> str:
    """Resolve model source path from Hugging Face id or ModelScope mirror."""
    if not use_modelscope:
        return model_name

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
    """Load tokenizer and CausalLM model with auto backend placement."""
    model_path = resolve_model_path(model_name=model_name, use_modelscope=use_modelscope, cache_dir=cache_dir)

    device = get_device_info()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=get_torch_dtype(device.device_type, prefer_bf16=prefer_bf16),
        device_map=get_device_map(device.device_type),
    )

    if device.device_type in ["npu", "mps"]:
        if hasattr(model, "to"):
            model_any: Any = model
            model_any.to(torch.device(device.device_type))

    return tokenizer, model, model_path, device


def build_zero_shot_prompt(title: str, review: str) -> str:
    """Build baseline prompt for Task A."""
    return f"""Given the following restaurant review, please rate it from 1 to 5 stars, where:
- 1 star: Very poor experience
- 2 stars: Poor experience
- 3 stars: Average experience
- 4 stars: Good experience
- 5 stars: Excellent experience

Title: {title}
Review: {review}

Please provide only a single number (1, 2, 3, 4, or 5) as your rating.
Rating: """


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

    lines = [
        "You are an expert restaurant reviewer. Rate each review strictly on a 1-5 integer scale.",
        "Follow the rubric: 1=awful, 2=poor, 3=average, 4=good, 5=excellent.",
        "Respond ONLY in the format `Rating: <digit>` with no extra text.",
        "",
        "Examples:",
    ]

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
    if title:
        lines.append(f"Title: {title}")
    lines.append(f"Review: {review}")
    lines.append("")
    lines.append("Return format: `Rating: <digit>` (no explanation).")
    lines.append("Rating: ")

    return "\n".join(lines)


def extract_rating_from_output(output_text: str, default_rating: int = 3) -> int:
    """Parse the first valid rating from model output."""
    numbers = re.findall(r"\b([1-5])\b", output_text)
    if numbers:
        return int(numbers[0])

    star_match = re.search(r"(\d+)\s*star", output_text, flags=re.IGNORECASE)
    if star_match:
        candidate = int(star_match.group(1))
        if 1 <= candidate <= 5:
            return candidate

    return default_rating


def _build_chat_inputs(
    tokenizer,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> Any:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer([text], return_tensors="pt")


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
    model_inputs = _build_chat_inputs(tokenizer, prompt, system_prompt=system_prompt)
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
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0].strip()


def predict_rating(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 10,
    system_prompt: Optional[str] = None,
) -> Tuple[str, int]:
    """Generate raw output and parsed rating in one call."""
    output = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=False,
        system_prompt=system_prompt,
    )
    return output, extract_rating_from_output(output)


def generate_rating_finetuned(
    model,
    tokenizer,
    title: str,
    review: str,
    max_new_tokens: int = 10,
) -> str:
    """Inference format aligned with LoRA instruction tuning data."""
    instruction = (
        "Given the following restaurant review, rate it from 1 to 5 stars. "
        "Respond ONLY with a single digit (1, 2, 3, 4, or 5). Do not include words or punctuation."
    )
    input_text = f"Title: {title}\nReview: {review}" if title else f"Review: {review}"
    prompt = f"{instruction}\n\n{input_text}\n\nRating (1-5):"

    return generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=False,
        system_prompt="Return only integers 1-5 with no explanation.",
    )


def load_merged_lora_model(
    base_model_name: str,
    lora_dir: str,
    use_modelscope: bool = False,
    cache_dir: str = "./models",
    prefer_bf16: bool = True,
):
    """Load base model, attach LoRA adapter, and merge weights for inference."""
    peft_module = import_module("peft")
    peft_model_class = getattr(peft_module, "PeftModel")

    tokenizer, base_model, _, device = load_tokenizer_and_model(
        model_name=base_model_name,
        use_modelscope=use_modelscope,
        cache_dir=cache_dir,
        prefer_bf16=prefer_bf16,
    )

    lora_model = peft_model_class.from_pretrained(base_model, lora_dir)
    merged_model = lora_model.merge_and_unload()

    if device.device_type in ["npu", "mps"]:
        merged_model = merged_model.to(device.device_type)

    return tokenizer, merged_model, device
