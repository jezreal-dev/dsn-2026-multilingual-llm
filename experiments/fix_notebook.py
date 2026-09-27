import ast
import json
from pathlib import Path

TARGET_FILES = [
    Path("/home/jmomoh/dsn-ai-bootcamp/kaggle_kernel/notebookdc2758c08b.ipynb"),
    Path("/home/jmomoh/dsn-ai-bootcamp/multilingual_topic_headline_dsn2026.ipynb"),
]

CELL_7_CODE = r"""# 7. Multi-Task PEFT / LoRA Fine-Tuning Pipeline
from datasets import Dataset

train_samples = []
for _, row in train_df.iterrows():
    # Task A sample: classify topic
    prompt_a = build_topic_prompt(row["text"], row["language"])
    target_a = f"{row['label']}<|im_end|>"
    train_samples.append({"prompt": prompt_a, "target": target_a})
    
    # Task B sample: generate headline conditioned on topic
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

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"""

CELL_8_CODE = r"""# 8. Execute LoRA Fine-Tuning (T4 GPU Memory Budgeted)
training_args = TrainingArguments(
    output_dir="./lora_qwen_dsn2026",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    warmup_steps=50,
    save_strategy="no",
    report_to="none"
)

# Custom Rectangular Causal Collator:
# Dynamically pads input_ids with pad_token_id and labels with -100 to the batch's max sequence length
class CausalDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features):
        batch_max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            ids = f["input_ids"]
            mask = f.get("attention_mask", [1] * len(ids))
            labels = f.get("labels", ids)
            pad_len = batch_max_len - len(ids)

            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long)
        }

data_collator = CausalDataCollator(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("Commencing PEFT Fine-Tuning...")
trainer.train()
print("Fine-tuning completed successfully!")
"""

def update_all():
    source_nb = TARGET_FILES[0]
    with open(source_nb, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Cell 7: Multi-task Tokenization with prompt loss masking (index 7)
    nb_data["cells"][7]["source"] = [line + "\n" for line in CELL_7_CODE.strip().splitlines()]
    # Cell 8: CausalDataCollator and Training (index 8)
    nb_data["cells"][8]["source"] = [line + "\n" for line in CELL_8_CODE.strip().splitlines()]

    # AST verify all code cells
    for idx, cell in enumerate(nb_data["cells"]):
        if cell["cell_type"] != "code":
            continue
        code_text = "".join(cell["source"])
        clean_code = "\n".join(l for l in code_text.splitlines() if not l.strip().startswith("!"))
        try:
            ast.parse(clean_code)
            print(f"Cell {idx} AST validation: PASSED")
        except SyntaxError as e:
            print(f"Cell {idx} FAILED AST at line {e.lineno}: {e.text}")
            raise e

    for path in TARGET_FILES:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=2)
        print(f"Updated {path}")

if __name__ == "__main__":
    update_all()
