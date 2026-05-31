"""PeerRead dataset for multi-task paper classification."""
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


DOMAIN_LABELS = {"NLP": 0, "CV": 1, "ML": 2, "AI": 3}
METHOD_LABELS = {"Empirical": 0, "Theoretical": 1, "Survey": 2, "Benchmark": 3}


class PeerReadDataset(Dataset):
    """Dataset wrapping a list of paper dicts for classification training."""

    def __init__(
        self,
        samples: List[Dict],
        tokenizer_name: str = "allenai/scibert_scivocab_uncased",
        max_length: int = 512,
    ):
        self.samples = samples
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Multi-hot domain labels
        domain_labels = torch.zeros(len(DOMAIN_LABELS))
        for d in sample.get("domains", []):
            if d in DOMAIN_LABELS:
                domain_labels[DOMAIN_LABELS[d]] = 1.0

        # Single-label method type
        method_label = METHOD_LABELS.get(sample.get("method_type", "Empirical"), 0)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "domain_labels": domain_labels,
            "method_label": method_label,
        }
