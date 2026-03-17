# QwenLora Restaurant Review Rating Project

## 1. 项目简介

本项目围绕课程任务实现了 3 条情感评分路线，目标是对餐厅评论做 1-5 星分类：

- Task A: Zero-shot 推理
- Task B.1.1: N-shot In-Context Learning
- Task B.1.2: LoRA 指令微调后推理

数据默认位于 `data/`，核心代码拆分为 `dataset.py`、`model.py`、`train.py`、`eval.py`。

## 2. 技术说明

- 模型框架: Transformers CausalLM
- 参数高效微调: PEFT (LoRA)
- 训练与评估: Hugging Face Trainer + sklearn 指标
- 设备适配: 自动检测 CUDA / NPU / MPS / CPU
- 推理输出解析: 正则抽取 1-5 星并提供回退策略

评估指标包含：

- Accuracy
- Macro-F1
- Weighted-F1
- Classification Report
- Confusion Matrix

## 3. 环境与依赖

当前仓库已合并为单一依赖文件：`requirements.txt`。

该文件按你的 GPU 环境（CUDA 12.8）配置了 PyTorch 源：

```txt
--extra-index-url https://download.pytorch.org/whl/cu128
torch>=2.4
...
```

安装方式：

```bash
pip install -r requirements.txt
```

## 4. 核心文件与函数说明

### dataset.py

- `read_csv_with_encoding(filepath, encodings=None)`:
  鲁棒读取 CSV，自动尝试多编码。
- `rating_to_numeric(rating_str)`:
  将 `"4 star"` 这类文本标签转换为整数标签。
- `add_numeric_rating_column(df, source_col="Rating", target_col="rating_numeric")`:
  为 DataFrame 添加数值标签列。
- `balanced_sample(df, label_col, sample_size, seed, min_items_per_class=1)`:
  采样时尽量保证类别覆盖，便于调试与快速实验。
- `load_project_data(data_dir="./data")`:
  一次性加载 train/test/answer 三份数据并处理标签列。
- `split_train_val(train_df, seed=5494, test_size=0.2, label_col="rating_numeric", debug_sample_size=None)`:
  分层切分训练/验证集，可选下采样。
- `build_example_library(train_df, n_per_rating=10, seed=5494, rating_col="rating_numeric")`:
  从训练集构建 few-shot 示例库。
- `prepare_instruction_data(df, rating_col="rating_numeric")`:
  将数据转成 SFT 指令样本 `instruction/input/output`。
- `select_instruction_subset(records, max_samples)`:
  选择部分样本用于快速实验。

### model.py

- `get_device_info()`:
  自动识别运行设备并返回设备类型、名称、显存信息。
- `get_device_map(device_type)`:
  按设备返回 `transformers` 的加载策略。
- `get_torch_dtype(device_type, prefer_bf16=True)`:
  按设备选择推理/训练 dtype。
- `resolve_model_path(model_name, use_modelscope=False, cache_dir="./models")`:
  解析模型来源（HF 或 ModelScope）。
- `load_tokenizer_and_model(...)`:
  统一加载 tokenizer 与基础模型。
- `build_zero_shot_prompt(title, review)`:
  生成 zero-shot 提示词。
- `build_nshot_prompt(title, review, examples, n=4, seed=None)`:
  生成 few-shot 提示词。
- `extract_rating_from_output(output_text, default_rating=3)`:
  从模型文本输出中解析评分。
- `generate_response(...)`:
  统一单轮生成接口。
- `predict_rating(model, tokenizer, prompt, ...)`:
  一次调用返回原始输出和评分。
- `generate_rating_finetuned(...)`:
  微调模型专用评分生成接口。
- `load_merged_lora_model(base_model_name, lora_dir, ...)`:
  加载基础模型 + LoRA 并执行 merge，用于高效推理。

### train.py

- `InstructionDataset(records, tokenizer, max_length=512)`:
  将指令样本转成训练所需张量格式。
- `LoraTrainingConfig`:
  LoRA 训练参数配置对象（路径、batch、学习率、LoRA 超参等）。
- `select_records(records, max_samples)`:
  训练前样本截断。
- `attach_lora_adapter(model, config)`:
  为基础模型挂载 LoRA。
- `build_training_arguments(config)`:
  构建 Hugging Face `TrainingArguments`。
- `train_lora_model(base_model, tokenizer, train_records, val_records, config=None)`:
  LoRA 微调主流程，训练后保存模型与 tokenizer。

### eval.py

- `run_inference_loop(data_df, predict_fn, id_col="Review_id", progress_every=50)`:
  通用推理循环，负责收集预测、耗时和调试信息。
- `run_zero_shot_inference(...)`:
  Task A 零样本推理。
- `run_nshot_inference(...)`:
  Task B.1.1 few-shot 推理。
- `run_lora_inference(...)`:
  Task B.1.2 LoRA 微调后推理。
- `evaluate_predictions(predictions_df, answer_df, ...)`:
  输出准确率、F1、报告和混淆矩阵。
- `save_predictions(predictions_df, output_path)`:
  导出预测 CSV。
- `build_comparison_table(rows)`:
  构建方法对比表。
- `time_per_sample(total_seconds, sample_count)`:
  计算单样本平均耗时。

## 5. 在 Notebook 中如何调用

下面给出推荐调用顺序（按实验流程）：

```python
from dataset import (
    load_project_data,
    split_train_val,
    build_example_library,
    prepare_instruction_data,
)
from model import load_tokenizer_and_model, load_merged_lora_model
from train import LoraTrainingConfig, train_lora_model
from eval import (
    run_zero_shot_inference,
    run_nshot_inference,
    run_lora_inference,
    evaluate_predictions,
)

# 1) 读数据
train_df, test_df, answer_df = load_project_data("./data")

# 2) 切分训练/验证
train_split, val_split = split_train_val(train_df, seed=5494, test_size=0.2)

# 3) 加载基础模型
tokenizer, base_model, model_path, device = load_tokenizer_and_model(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    use_modelscope=False,
)

# 4) Zero-shot
zero_out = run_zero_shot_inference(base_model, tokenizer, test_df)
zero_metric = evaluate_predictions(zero_out["predictions_df"], answer_df)

# 5) N-shot
example_lib = build_example_library(train_split, n_per_rating=10, seed=5494)
nshot_out = run_nshot_inference(base_model, tokenizer, test_df, example_lib, n_shot=4)
nshot_metric = evaluate_predictions(nshot_out["predictions_df"], answer_df)

# 6) LoRA 训练
train_records = prepare_instruction_data(train_split)
val_records = prepare_instruction_data(val_split)
cfg = LoraTrainingConfig(output_dir="./qwen_lora_checkpoints", final_dir="./qwen_lora_final")
train_res = train_lora_model(base_model, tokenizer, train_records, val_records, cfg)

# 7) LoRA 推理（merge 后）
tok_lora, merged_model, device = load_merged_lora_model(
    base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    lora_dir=train_res["output_dir"],
)
lora_out = run_lora_inference(merged_model, tok_lora, test_df)
lora_metric = evaluate_predictions(lora_out["predictions_df"], answer_df)
```

## 6. 目录说明

- `dataset.py`: 数据读取、标签处理、划分、指令样本构造
- `model.py`: 模型加载、prompt 构建、生成与评分解析
- `train.py`: LoRA 训练流程
- `eval.py`: 推理与评估流程
- `requirements.txt`: 单一依赖文件（GPU/CUDA 12.8）
