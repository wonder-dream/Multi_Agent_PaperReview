"""
摘要数据集: SciTLDR 极端摘要数据加载

SciTLDR 格式:
    source: List[str]     - 源文本句子列表
    target: List[str]     - 摘要句子列表
    source_labels: List[int] - 1=该句属于摘要 (用于抽取式)
"""
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List


class SciTLDRDataset(Dataset):
    """SciTLDR 摘要数据集，支持抽取式和生成式两种模式"""

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "facebook/bart-base",
        max_source_length: int = 512,
        max_target_length: int = 128,
        mode: str = "generative"
    ):
        """
        Args:
            data_path: SciTLDR JSONL 路径
            tokenizer_name: BART tokenizer
            max_source_length: 源文本最大长度
            max_target_length: 目标摘要最大长度
            mode: "generative" (Seq2Seq) 或 "extractive" (句子分类)
        """
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.samples = self._load_data(data_path)
        print(f"[SciTLDRDataset] 加载了 {len(self.samples)} 条样本 ({mode}) from {data_path}")

    def _load_data(self, data_path: str) -> List[Dict]:
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                source = item.get("source", [])
                target = item.get("target", [])
                if not source or not target:
                    continue
                samples.append(item)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        source_text = " ".join(sample["source"])
        target_text = " ".join(sample["target"])

        if self.mode == "generative":
            encoding = self.tokenizer(
                source_text, max_length=self.max_source_length,
                padding="max_length", truncation=True, return_tensors="pt"
            )
            target_encoding = self.tokenizer(
                target_text, max_length=self.max_target_length,
                padding="max_length", truncation=True, return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": target_encoding["input_ids"].squeeze(0),
            }
        else:
            # Extractive: 每个句子是一个分类样本
            encoding = self.tokenizer(
                source_text, max_length=self.max_source_length,
                padding="max_length", truncation=True, return_tensors="pt"
            )
            labels = sample.get("source_labels", [0] * len(sample["source"]))
            labels = labels[:self.max_source_length] + [0] * max(0, self.max_source_length - len(labels))
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(labels[:self.max_source_length], dtype=torch.float)
            }
