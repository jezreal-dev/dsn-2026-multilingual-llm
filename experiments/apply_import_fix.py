import json

notebook_path = '/home/jmomoh/dsn-ai-bootcamp/kaggle_kernel/notebookdc2758c08b.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# CELL 1: Environment Setup with explicit datasets import
cell_1_code = """# 1. Environment Setup & Dependency Installation
import os
import sys

# Install required competition libraries if running in Kaggle / Colab
!pip uninstall -y -q torchao
!pip install -q peft evaluate rouge-score bert-score datasets

import time
import re
import csv
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import f1_score
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# Set deterministic seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

START_TIME = time.time()
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
"""

# CELL 7: Multi-Task LoRA Setup with explicit Dataset import
cell_7_code = """# 7. Multi-Task PEFT / LoRA Fine-Tuning Pipeline (r=32 Capacity Expansion)
from datasets import Dataset

train_samples = []

for _, row in train_df.iterrows():
    # Task A: Topic Classification
    prompt_a = build_topic_prompt(row["text"], row["language"])
    target_a = f"{row['label']}<|im_end|>"
    train_samples.append({"prompt": prompt_a, "target": target_a})
    
    # Task B: Headline Generation (conditioned on ground truth topic for training)
    prompt_b = build_headline_prompt(row["text"], row["language"], row["label"])
    target_b = f"{row['headline']}<|im_end|>"
    train_samples.append({"prompt": prompt_b, "target": target_b})

print(f"Compiled {len(train_samples)} multi-task instruction instances.")
train_dataset = Dataset.from_list(train_samples)

def tokenize_func(batch):
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    
    for prompt, target in zip(batch["prompt"], batch["target"]):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        
        full_ids = prompt_ids + target_ids
        if len(full_ids) > 512:
            full_ids = full_ids[:512]
            
        # Mask prompt tokens (-100) so loss is ONLY computed on the target generation
        labels = [-100] * len(prompt_ids) + target_ids
        if len(labels) > 512:
            labels = labels[:512]
            
        batch_input_ids.append(full_ids)
        batch_attention_mask.append([1] * len(full_ids))
        batch_labels.append(labels)
        
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }

tokenized_dataset = train_dataset.map(tokenize_func, batched=True, remove_columns=["prompt", "target"])

# Expanded LoRA rank (r=32, alpha=64) across all linear projection modules
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"""

def to_notebook_source(code_str):
    lines = code_str.split('\n')
    return [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

nb['cells'][1]['source'] = to_notebook_source(cell_1_code)
nb['cells'][7]['source'] = to_notebook_source(cell_7_code)

# Clear outputs
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Dataset import fix successfully applied!")
