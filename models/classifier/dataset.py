"""
分类器数据集定义
支持三种数据加载模式:
    1. DomainDataset: 领域分类 (NLP, CV, ML, AI)
    2. QualityDataset: 质量分类 (accept, reject)
    3. MultiTaskDataset: 多任务联合训练 (需要同时有domain和quality标签)

数据格式:
    所有数据集都读取JSONL文件，每条记录包含:
    - text: "Title: xxx Abstract: xxx"
    - label: 标签字符串
    - task: "domain" 或 "quality"
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Tuple
import os


class DomainDataset(Dataset):
    """
    领域分类数据集
    
    从JSONL文件加载，支持以下数据源:
        - arxiv_train.jsonl (单独arXiv)
        - merged_train.jsonl (arXiv + PeerRead合并)
    
    标签映射:
        NLP -> 0, CV -> 1, ML -> 2, AI -> 3
    
    Example:
        >>> dataset = DomainDataset("processed_data/classification_arxiv/arxiv_train.jsonl")
        >>> print(len(dataset))  # 20000
        >>> sample = dataset[0]
        >>> print(sample.keys())  # dict_keys(['input_ids', 'attention_mask', 'labels'])
    """
    
    LABEL2ID = {"NLP": 0, "CV": 1, "ML": 2, "AI": 3}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "allenai/scibert_scivocab_uncased",
        max_length: int = 512
    ):
        """
        Args:
            data_path: JSONL文件路径
            tokenizer_name: SciBERT tokenizer名称
            max_length: 最大序列长度
        """
        self.data_path = data_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 加载数据
        self.samples = self._load_data()
        print(f"[DomainDataset] 加载了 {len(self.samples)} 条样本 from {data_path}")
    
    def _load_data(self) -> List[Dict]:
        """从JSONL文件加载数据"""
        samples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                text = item.get("text", "")
                label = item.get("label", "")
                
                # 过滤无效标签
                if label not in self.LABEL2ID:
                    continue
                
                samples.append({
                    "text": text,
                    "label": self.LABEL2ID[label],
                    "source": item.get("source", "unknown")
                })
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            字典包含:
                - input_ids: (seq_len,)
                - attention_mask: (seq_len,)
                - labels: scalar tensor
        """
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["label"], dtype=torch.long)
        }
    
    def get_label_distribution(self) -> Dict[str, int]:
        """获取标签分布统计"""
        dist = {}
        for sample in self.samples:
            label_name = self.ID2LABEL[sample["label"]]
            dist[label_name] = dist.get(label_name, 0) + 1
        return dist


class QualityDataset(Dataset):
    """
    质量分类数据集
    
    从JSONL文件加载，数据源:
        - quality_train.jsonl (仅PeerRead)
    
    标签映射:
        accept -> 0, reject -> 1
    
    注意类别不平衡: accept通常占70-80%
    使用 get_class_weights() 获取用于WeightedLoss的权重
    
    Example:
        >>> dataset = QualityDataset("processed_data/arxiv_PeerRead_merge_data/classification/quality_train.jsonl")
        >>> weights = dataset.get_class_weights()  # tensor([1.0, 3.2])
    """
    
    LABEL2ID = {"accept": 0, "reject": 1}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "allenai/scibert_scivocab_uncased",
        max_length: int = 512
    ):
        """
        Args:
            data_path: JSONL文件路径
            tokenizer_name: SciBERT tokenizer名称
            max_length: 最大序列长度
        """
        self.data_path = data_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.samples = self._load_data()
        print(f"[QualityDataset] 加载了 {len(self.samples)} 条样本 from {data_path}")
    
    def _load_data(self) -> List[Dict]:
        """从JSONL文件加载数据"""
        samples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                text = item.get("text", "")
                label = item.get("label", "")
                
                if label not in self.LABEL2ID:
                    continue
                
                samples.append({
                    "text": text,
                    "label": self.LABEL2ID[label],
                    "source": item.get("source", "unknown")
                })
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["label"], dtype=torch.long)
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        计算类别权重 (用于处理不平衡)
        
        权重计算方式: weight_i = total_samples / (num_classes * count_i)
        
        Returns:
            权重张量，shape (2,)
        """
        label_counts = [0, 0]
        for sample in self.samples:
            label_counts[sample["label"]] += 1
        
        total = sum(label_counts)
        weights = [total / (len(label_counts) * count) if count > 0 else 1.0 
                   for count in label_counts]
        
        print(f"[QualityDataset] 类别分布: accept={label_counts[0]}, reject={label_counts[1]}")
        print(f"[QualityDataset] 类别权重: {weights}")
        
        return torch.tensor(weights, dtype=torch.float)
    
    def get_label_distribution(self) -> Dict[str, int]:
        """获取标签分布统计"""
        dist = {}
        for sample in self.samples:
            label_name = self.ID2LABEL[sample["label"]]
            dist[label_name] = dist.get(label_name, 0) + 1
        return dist


class MultiTaskDataset(Dataset):
    """
    多任务联合训练数据集
    
    同时加载domain和quality数据，每条样本共享同一篇论文的文本，
    但有两个标签: domain_label 和 quality_label。
    
    数据组织方式:
        - domain数据: merged_*.jsonl (arXiv + PeerRead)
        - quality数据: quality_*.jsonl (仅PeerRead)
    
    由于quality数据只有PeerRead的论文有，而domain数据包含arXiv论文
    (arXiv无quality标签)，因此采用以下策略:
        1. 对同时有domain和quality标签的论文 (PeerRead部分)，使用真实标签
        2. 对只有domain标签的论文 (arXiv部分)，quality标签设为-1 (忽略损失)
    
    实际实现：从两个文件分别加载，每个step返回一个domain样本和一个quality样本
    通过交替采样实现"联合训练"的效果
    
    Example:
        >>> dataset = MultiTaskDataset(
        ...     domain_path="merged_train.jsonl",
        ...     quality_path="quality_train.jsonl"
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())  # dict_keys(['input_ids', 'attention_mask', 'domain_labels', 'quality_labels'])
    """
    
    DOMAIN_LABEL2ID = {"NLP": 0, "CV": 1, "ML": 2, "AI": 3}
    QUALITY_LABEL2ID = {"accept": 0, "reject": 1}
    
    def __init__(
        self,
        domain_data_path: str,
        quality_data_path: str,
        tokenizer_name: str = "allenai/scibert_scivocab_uncased",
        max_length: int = 512,
        oversample_quality: bool = True
    ):
        """
        Args:
            domain_data_path: Domain分类JSONL路径
            quality_data_path: Quality分类JSONL路径
            tokenizer_name: SciBERT tokenizer
            max_length: 最大序列长度
            oversample_quality: 是否对quality样本进行过采样以平衡两个任务的样本数
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.oversample_quality = oversample_quality

        # 加载两个任务的数据
        self.domain_samples = self._load_domain_data(domain_data_path)
        self.quality_samples = self._load_quality_data(quality_data_path)

        # 计算有效长度
        self.domain_len = len(self.domain_samples)
        self.quality_len = len(self.quality_samples)

        if self.domain_len == 0 and self.quality_len == 0:
            raise ValueError(
                f"[MultiTaskDataset] Domain和Quality数据均为空! "
                f"请检查数据文件:\n  domain: {domain_data_path}\n  quality: {quality_data_path}"
            )

        if self.quality_len == 0:
            print("[MultiTaskDataset] WARNING: Quality数据为空，将仅使用Domain样本训练")

        if oversample_quality and self.quality_len > 0 and self.quality_len < self.domain_len:
            # 对quality进行过采样
            import random
            random.seed(42)
            extra_needed = self.domain_len - self.quality_len
            extra_samples = random.choices(self.quality_samples, k=extra_needed)
            self.quality_samples.extend(extra_samples)
            self.quality_len = len(self.quality_samples)
            print(f"[MultiTaskDataset] Quality过采样后: {self.quality_len} 样本")

        # 总长度: quality为空时只用domain，否则交替采样取两倍较大值
        if self.quality_len == 0:
            self._length = self.domain_len
        else:
            self._length = max(self.domain_len, self.quality_len)

        print(f"[MultiTaskDataset] Domain: {self.domain_len}, Quality: {self.quality_len}")
        print(f"[MultiTaskDataset] 总有效长度: {self._length}")
    
    def _load_domain_data(self, path: str) -> List[Dict]:
        """加载Domain数据"""
        samples = []
        if not os.path.exists(path):
            print(f"[MultiTaskDataset] WARNING: Domain数据文件不存在: {path}")
            return samples
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                text = item.get("text", "")
                label = item.get("label", "")
                if label in self.DOMAIN_LABEL2ID:
                    samples.append({
                        "text": text,
                        "domain_label": self.DOMAIN_LABEL2ID[label],
                        # quality标签设为-1表示忽略
                        "quality_label": -1,
                        "has_quality": False
                    })
        return samples
    
    def _load_quality_data(self, path: str) -> List[Dict]:
        """加载Quality数据"""
        samples = []
        if not os.path.exists(path):
            print(f"[MultiTaskDataset] WARNING: Quality数据文件不存在: {path}")
            return samples
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                text = item.get("text", "")
                label = item.get("label", "")
                if label in self.QUALITY_LABEL2ID:
                    # 尝试从quality数据推断domain标签
                    # 如果source中包含venue信息，可以映射到domain
                    source = item.get("source", "")
                    inferred_domain = self._infer_domain_from_source(source)

                    samples.append({
                        "text": text,
                        "domain_label": inferred_domain if inferred_domain is not None else -1,
                        "quality_label": self.QUALITY_LABEL2ID[label],
                        "has_quality": True
                    })
        return samples
    
    def _infer_domain_from_source(self, source: str) -> Optional[int]:
        """
        从source/venue推断domain标签
        
        PeerRead venue映射:
            acl, conll -> NLP
            iclr, nips -> ML
            arxiv.cs.ai -> AI
        """
        source_lower = source.lower()
        if any(k in source_lower for k in ["acl", "conll", "cs.cl"]):
            return self.DOMAIN_LABEL2ID["NLP"]
        elif any(k in source_lower for k in ["iclr", "nips", "cs.lg"]):
            return self.DOMAIN_LABEL2ID["ML"]
        elif "cs.ai" in source_lower:
            return self.DOMAIN_LABEL2ID["AI"]
        elif "cs.cv" in source_lower:
            return self.DOMAIN_LABEL2ID["CV"]
        return None
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取样本

        交替返回domain样本和quality样本:
            - 偶数idx: 返回domain样本 (quality_label=-1)
            - 奇数idx: 返回quality样本 (domain_label可能为-1)
        Quality数据为空时，全部返回domain样本。
        """
        if self.quality_len == 0 or idx % 2 == 0:
            # Domain样本
            sample_idx = idx % self.domain_len
            sample = self.domain_samples[sample_idx]
        else:
            # Quality样本
            sample_idx = (idx // 2) % self.quality_len
            sample = self.quality_samples[sample_idx]
        
        encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "domain_labels": torch.tensor(sample["domain_label"], dtype=torch.long),
            "quality_labels": torch.tensor(sample["quality_label"], dtype=torch.long)
        }
        
        return result
    
    def get_quality_class_weights(self) -> torch.Tensor:
        """计算Quality任务的类别权重"""
        label_counts = [0, 0]
        for sample in self.quality_samples:
            if sample["quality_label"] >= 0:
                label_counts[sample["quality_label"]] += 1

        total = sum(label_counts)
        if total == 0:
            print("[MultiTaskDataset] Quality数据为空，返回均匀权重")
            return torch.tensor([1.0, 1.0], dtype=torch.float)

        weights = [total / (len(label_counts) * count) if count > 0 else 1.0
                   for count in label_counts]

        print(f"[MultiTaskDataset] Quality类别权重: {weights}")
        return torch.tensor(weights, dtype=torch.float)
    
    def get_domain_class_weights(self) -> torch.Tensor:
        """计算Domain任务的类别权重"""
        label_counts = [0, 0, 0, 0]
        for sample in self.domain_samples:
            label_counts[sample["domain_label"]] += 1
        
        total = sum(label_counts)
        weights = [total / (len(label_counts) * count) if count > 0 else 1.0
                   for count in label_counts]
        
        print(f"[MultiTaskDataset] Domain类别权重: {weights}")
        return torch.tensor(weights, dtype=torch.float)


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """
    创建DataLoader的便捷函数
    
    Args:
        dataset: 数据集实例
        batch_size: 批量大小
        shuffle: 是否打乱
        num_workers: 数据加载线程数
    
    Returns:
        DataLoader实例
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
