# 合并数据集说明 (ArXiv + PeerRead)

由 `merge_datasets.py` 自动合并产出，包含分类训练集与检索语料库。

## 输出结构

```
processed_data/arxiv_PeerRead_merge_data
├── classification/
│   ├── merged_train.jsonl      # domain 分类 (arxiv + peerread)
│   ├── merged_dev.jsonl
│   ├── merged_test.jsonl
│   ├── quality_train.jsonl     # quality 分类 (仅 peerread)
│   ├── quality_dev.jsonl
│   └── quality_test.jsonl
└── retrieval/
    ├── corpus_merged.jsonl         # 检索语料库 (全部 arxiv + 全部 peerread)
    ├── merged_train_pairs.jsonl    # 对比学习三元组
    ├── merged_dev_pairs.jsonl
    └── merged_test_pairs.jsonl
```

---

## 1. 分类数据

### 1.1 Domain 分类 (merged_*.jsonl)

**ArXiv 采样策略**：每类最多 5,000 篇，避免 50 万原始数据淹没 PeerRead。

```json
{"text": "Title:...
Abstract:...", "label": "NLP", "task": "domain", "source": "arxiv"}
{"text": "Title:...
Abstract:...", "label": "NLP", "task": "domain", "source": "acl_2017"}
```

| 来源 | train 数量 | 说明 |
|:---|:---|:---|
| arXiv (采样) | ~3,500/类 | 预印本，标签干净 |
| PeerRead | ~200/类 | 会议论文，标签来自 venue 映射 |
| **合计** | ~15,000 | 4 类 (NLP/CV/ML/AI) |

**用途**：训练 **SciBERT 领域分类器**，或作为多任务学习的 domain 分支。

### 1.2 Quality 分类 (quality_*.jsonl)

```json
{"text": "Title:...
Abstract:...", "label": "accept", "task": "quality", "source": "iclr_2017"}
```

- **仅来自 PeerRead**，arXiv 无 accept/reject 标签
- **2 类**：`accept` / `reject`
- 注意类别不平衡（accept 通常占 70-80%），训练时建议加 `class_weight`

**用途**：训练 **SciBERT 质量分类器**，或作为多任务学习的 quality 分支。

---

## 2. 检索数据

### 2.1 语料库 (corpus_merged.jsonl)

```json
{"id": "arxiv_1810.04805", "text": "Title:...
Abstract:...", "domain": "NLP", "source": "arxiv", "venue": "cs.CL"}
{"id": "peerread_acl_2017_1234567", "text": "Title:...
Abstract:...", "domain": "NLP", "source": "peerread", "venue": "acl_2017"}
```

| 来源 | 数量 | 说明 |
|:---|:---|:---|
| arXiv (全部) | ~50-60 万 | 700MB 筛选结果全保留 |
| PeerRead | ~1.5 万 | 会议论文 |
| **合计** | **~55 万+** | 覆盖 NLP/CV/ML/AI |

**用途**：构建 **FAISS 语义索引**，作为检索系统的文档库。

### 2.2 对比学习训练对 (merged_*_pairs.jsonl)

```json
{
  "anchor_text": "论文A摘要...",
  "positive_text": "同领域论文B摘要...",
  "negative_text": "不同领域论文C摘要...",
  "domain": "NLP"
}
```

- **正样本**：同 domain 随机采样
- **负样本**：不同 domain 随机采样
- 每类最多 5,000 对，防止文件过大

**用途**：**微调 SPECTER** 等语义编码器（拉近同领域、推远不同领域）。

---

## 使用建议

| 模块 | 推荐数据 | 说明 |
|:---|:---|:---|
| **Domain 分类** | `merged_train.jsonl` | 多源混合，标签均衡 |
| **Quality 分类** | `quality_train.jsonl` | 仅 PeerRead，注意不平衡 |
| **多任务联合训练** | 上面两个 concat | 共享 SciBERT 编码器，双头输出 |
| **语义检索 (Inference)** | `corpus_merged.jsonl` | 55 万+ 论文，覆盖广 |
| **语义检索 (Fine-tune)** | `merged_train_pairs.jsonl` | 对比学习微调 SPECTER |

---

## 规模速查

| 数据集 | train | dev | test | 备注 |
|:---|:---|:---|:---|:---|
| merged (domain) | ~12,000 | ~1,500 | ~1,500 | arxiv+peerread |
| quality | ~8,000 | ~500 | ~500 | 仅 peerread |
| corpus | 55 万+ | - | - | 检索库 |
| pairs | ~18,000 | ~1,000 | ~1,000 | 对比学习 |
