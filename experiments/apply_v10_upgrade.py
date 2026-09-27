import json

notebook_path = '/home/jmomoh/dsn-ai-bootcamp/kaggle_kernel/notebookdc2758c08b.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# CELL 6: Clean Inverted Prompt with Language Category Mapping & max_new_tokens=48
cell_6_code = """# 6. Prompt Engineering & Vectorized Batched Generation (Version 10 Refinements)
# 1. Full language names outperform ISO codes.
# 2. Sequential decomposition: Topic -> Headline conditioned on Topic.
# 3. Language-Specific Category Mapping eliminates out-of-domain category leakage.
# 4. Full article context (800 chars) passed to headline generator.
# 5. Soft repetition_penalty=1.15 WITHOUT hard ngram blocking, preserving African language grammar.
# 6. Optimized token ceiling (48 tokens) prevents truncation while avoiding run-on sentences.

LANGUAGE_CATEGORY_MAP = {
    "hau": ["business", "entertainment", "health", "politics", "religion", "sports", "technology"],
    "ibo": ["business", "entertainment", "health", "politics", "religion", "sports"],
    "pcm": ["business", "entertainment", "health", "politics", "sports"],
    "yor": ["entertainment", "health", "politics", "religion", "sports"]
}

def build_topic_prompt(text: str, language: str) -> str:
    lang_full = LANG_NAME_MAP.get(language, language)
    valid_cats = LANGUAGE_CATEGORY_MAP.get(language, VALID_LABELS)
    cats_str = ", ".join(valid_cats)
    truncated_text = text[:800]
    return (
        f"<|im_start|>system\\nYou are an expert multilingual news editor specializing in African languages.<|im_end|>\\n"
        f"<|im_start|>user\\nClassify the following {lang_full} news article into exactly one of these categories: {cats_str}.\\n\\n"
        f"Article:\\n{truncated_text}\\n\\n"
        f"Respond with only the category name in lowercase.<|im_end|>\\n"
        f"<|im_start|>assistant\\n"
    )

def build_headline_prompt(text: str, language: str, topic: str) -> str:
    lang_full = LANG_NAME_MAP.get(language, language)
    truncated_text = text[:800]
    return (
        f"<|im_start|>system\\nYou are an expert news editor in {lang_full}.<|im_end|>\\n"
        f"<|im_start|>user\\nWrite a concise, accurate headline in {lang_full} for this {topic} article.\\n\\n"
        f"Article:\\n{truncated_text}\\n\\n"
        f"Headline:<|im_end|>\\n"
        f"<|im_start|>assistant\\n"
    )

def parse_topic_output(raw_text: str, language: str = None) -> str:
    cleaned = raw_text.strip().lower()
    valid_cats = LANGUAGE_CATEGORY_MAP.get(language, VALID_LABELS) if language else VALID_LABELS
    for label in valid_cats:
        if label in cleaned:
            return label
    return "politics" if "politics" in valid_cats else valid_cats[0]

def batched_predict_topics(model, tokenizer, texts, languages, batch_size=16):
    \"\"\"Batched topic prediction with left-padding and greedy decoding.\"\"\"
    model.eval()
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    pred_topics = []
    prompts = [build_topic_prompt(t, l) for t, l in zip(texts, languages)]
    total = len(prompts)
    
    for i in range(0, total, batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_langs = languages[i : i + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        prompt_len = encoded.input_ids.shape[1]
        for idx_in_batch, out in enumerate(outputs):
            gen_text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            lang = batch_langs[idx_in_batch]
            pred_topics.append(parse_topic_output(gen_text, language=lang))
            
    tokenizer.padding_side = orig_padding_side
    return pred_topics

def batched_predict_headlines(model, tokenizer, texts, languages, topics, batch_size=16):
    \"\"\"Batched headline generation with soft repetition penalty and optimal token budget.\"\"\"
    model.eval()
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    pred_headlines = []
    prompts = [build_headline_prompt(t, l, top) for t, l, top in zip(texts, languages, topics)]
    total = len(prompts)
    
    for i in range(0, total, batch_size):
        batch_prompts = prompts[i : i + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=48,
                min_new_tokens=6,
                repetition_penalty=1.15,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        prompt_len = encoded.input_ids.shape[1]
        for idx_in_batch, out in enumerate(outputs):
            gen_text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True).strip()
            if not gen_text:
                orig_idx = i + idx_in_batch
                gen_text = texts[orig_idx][:50]
            pred_headlines.append(gen_text)
            
    tokenizer.padding_side = orig_padding_side
    return pred_headlines

def evaluate_dev_set(model, tokenizer, eval_df, batch_size=16):
    \"\"\"Evaluate complete dev dataframe using high-speed batched inference.\"\"\"
    texts = eval_df["text"].tolist()
    langs = eval_df["language"].tolist()
    
    print(f"Evaluating {len(eval_df)} samples: Step 1 (Topics)...")
    pred_topics = batched_predict_topics(model, tokenizer, texts, langs, batch_size=batch_size)
    
    print(f"Evaluating {len(eval_df)} samples: Step 2 (Headlines)...")
    pred_headlines = batched_predict_headlines(model, tokenizer, texts, langs, pred_topics, batch_size=batch_size)
    
    metrics = compute_evaluation_metrics(eval_df, pred_topics, pred_headlines)
    return pred_topics, pred_headlines, metrics

# Run complete zero-shot baseline benchmark across full dev set (869 samples)
print("Running complete zero-shot baseline benchmark on all dev samples...")
base_topics, base_headlines, baseline_metrics = evaluate_dev_set(model, tokenizer, dev_df, batch_size=16)

print("\\n=== ZERO-SHOT BASELINE BENCHMARK (FULL DEV SET) ===")
print(f"Baseline Macro-F1:     {baseline_metrics['Overall Macro-F1']:.4f}")
print(f"Baseline GenScore:     {baseline_metrics['Overall GenScore']:.4f}")
print(f"Baseline Merged Score: {baseline_metrics['Overall Merged Score']:.4f}")
"""

# CELL 7: Multi-Task PEFT / LoRA Fine-Tuning Pipeline (Standard 12,136 samples)
cell_7_code = """# 7. Multi-Task PEFT / LoRA Fine-Tuning Pipeline (r=32 Capacity)
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

# CELL 8: Execute 2-Epoch LoRA Fine-Tuning with Cosine Schedule
cell_8_code = """# 8. Execute LoRA Fine-Tuning (2 Epochs, Cosine Schedule)
training_args = TrainingArguments(
    output_dir="./lora_qwen_dsn2026",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1.8e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=2,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    warmup_ratio=0.05,
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

print("Commencing PEFT Fine-Tuning (2 Epochs)...")
trainer.train()
print("Fine-tuning completed successfully!")
"""

def to_notebook_source(code_str):
    lines = code_str.split('\n')
    return [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

nb['cells'][6]['source'] = to_notebook_source(cell_6_code)
nb['cells'][7]['source'] = to_notebook_source(cell_7_code)
nb['cells'][8]['source'] = to_notebook_source(cell_8_code)

# Clear outputs
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Version 10 updates successfully written to notebook!")
