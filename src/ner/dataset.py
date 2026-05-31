"""SciERC dataset for scientific NER training."""
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


ENTITY_TYPES = ["MODEL", "DATASET", "METRIC", "METHOD", "TASK"]

# BIO label mapping
LABEL2ID = {"O": 0}
for t in ENTITY_TYPES:
    LABEL2ID[f"B-{t}"] = len(LABEL2ID)
    LABEL2ID[f"I-{t}"] = len(LABEL2ID)
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class SciERCDataset(Dataset):
    """Dataset for SciERC-format samples with token-level BIO labels."""

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
        tokens = sample["tokens"]
        entities = sample.get("entities", [])

        # Build word-level BIO labels
        word_labels = ["O"] * len(tokens)
        for ent in entities:
            start, end = ent["start"], ent["end"]
            etype = ent["type"]
            if start < len(tokens):
                word_labels[start] = f"B-{etype}"
                for i in range(start + 1, min(end, len(tokens))):
                    word_labels[i] = f"I-{etype}"

        # Tokenize with subword alignment
        encoding = self.tokenizer(
            tokens, is_split_into_words=True,
            max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        word_ids = encoding.word_ids()
        label_ids = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_word:
                label_ids.append(LABEL2ID.get(word_labels[wid], 0))
            else:
                # Subword continuation: same label for I-*, else -100
                if word_labels[wid].startswith("I-"):
                    label_ids.append(LABEL2ID.get(word_labels[wid], -100))
                else:
                    label_ids.append(-100)
            prev_word = wid

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
