"""
信息抽取数据集: NER (BIO标注) + RE (实体对关系分类)
"""
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict


class NERDataset(Dataset):
    """BIO序列标注数据集，支持SciBERT子词对齐"""

    LABELS = ["O", "B-TASK", "I-TASK", "B-METHOD", "I-METHOD",
              "B-METRIC", "I-METRIC", "B-DATASET", "I-DATASET",
              "B-TERM", "I-TERM", "B-GENERIC", "I-GENERIC"]
    LABEL2ID = {l: i for i, l in enumerate(LABELS)}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}

    def __init__(self, data_path: str, tokenizer_name: str = "allenai/scibert_scivocab_uncased",
                 max_length: int = 256):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.samples = self._load_data(data_path)
        print(f"[NERDataset] 加载了 {len(self.samples)} 条样本 from {data_path}")

    def _load_data(self, data_path: str) -> List[Dict]:
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if not item.get("tokens"):
                    continue
                samples.append(item)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        tokens = sample["tokens"]
        raw_labels = sample["labels"]

        encoding = self.tokenizer(
            tokens, is_split_into_words=True,
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        word_ids = encoding.word_ids()

        aligned_labels = []
        prev_word_id = None
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                aligned_labels.append(0)  # 特殊token和PAD统一用O标签
            elif word_id != prev_word_id:
                aligned_labels.append(self.LABEL2ID.get(raw_labels[word_id], 0))
            else:
                aligned_labels.append(self.LABEL2ID.get(raw_labels[prev_word_id], 0))
            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }


SPECIAL_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]


class REDataset(Dataset):
    """关系分类数据集，使用实体标记构造输入"""

    RELATION_TYPES = ["NO-RELATION", "USED-FOR", "FEATURE-OF", "HYPONYM-OF",
                      "PART-OF", "COMPARE-WITH", "CONJUNCTION", "EVALUATED-ON"]
    RELATION2ID = {r: i for i, r in enumerate(RELATION_TYPES)}
    ID2RELATION = {v: k for k, v in RELATION2ID.items()}

    def __init__(self, data_path: str, tokenizer_name: str = "allenai/scibert_scivocab_uncased",
                 max_length: int = 256):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        self.samples = self._load_data(data_path)
        print(f"[REDataset] 加载了 {len(self.samples)} 条样本 from {data_path}")

    def _load_data(self, data_path: str) -> List[Dict]:
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                samples.append(item)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        sentence = sample["sentence"]
        e1_start, e1_end = sample["entity1_start"], sample["entity1_end"]
        e2_start, e2_end = sample["entity2_start"], sample["entity2_end"]

        words = sentence.split()
        e1_first = e1_start <= e2_start
        if e1_first:
            marked = (
                words[:e1_start] + ["[E1]"] + words[e1_start:e1_end + 1] + ["[/E1]"]
                + words[e1_end + 1:e2_start] + ["[E2]"] + words[e2_start:e2_end + 1] + ["[/E2]"]
                + words[e2_end + 1:]
            )
        else:
            marked = (
                words[:e2_start] + ["[E2]"] + words[e2_start:e2_end + 1] + ["[/E2]"]
                + words[e2_end + 1:e1_start] + ["[E1]"] + words[e1_start:e1_end + 1] + ["[/E1]"]
                + words[e1_end + 1:]
            )

        marked_sent = " ".join(marked)
        encoding = self.tokenizer(
            marked_sent, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )

        relation = sample.get("relation", "NO-RELATION")
        label = self.RELATION2ID.get(relation, 0)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

    def get_label_distribution(self) -> Dict[str, int]:
        dist = {}
        for sample in self.samples:
            rel = sample.get("relation", "NO-RELATION")
            dist[rel] = dist.get(rel, 0) + 1
        return dist
