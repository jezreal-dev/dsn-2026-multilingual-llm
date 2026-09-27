import torch

class CausalDataCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

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

if __name__ == "__main__":
    collator = CausalDataCollator(pad_token_id=0)
    # Test with differing sequence lengths: 368 and 430 (exact numbers from Kaggle crash)
    features = [
        {"input_ids": list(range(368)), "attention_mask": [1] * 368, "labels": [-100] * 300 + list(range(68))},
        {"input_ids": list(range(430)), "attention_mask": [1] * 430, "labels": [-100] * 350 + list(range(80))}
    ]
    batch = collator(features)
    print("Collation success!")
    print(f"input_ids shape: {batch['input_ids'].shape} (Expected: [2, 430])")
    print(f"labels shape:    {batch['labels'].shape} (Expected: [2, 430])")
    assert batch['input_ids'].shape == (2, 430)
    assert batch['labels'].shape == (2, 430)
    assert (batch['labels'][0, 368:] == -100).all()
    print("ALL ASSERTIONS PASSED!")
