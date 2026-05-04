# PeerRead 清洗数据说明

## 输出文件

```
processed_data/PeerRead_processed_data
├── classification/           # 文本分类 (domain + quality)
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── summarization/            # 审稿意见生成
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
└── retrieval/                # 语义检索
    ├── corpus.jsonl
    ├── train_pairs.jsonl
    ├── dev_pairs.jsonl
    └── test_pairs.jsonl
```

---

## 1. 分类数据 (classification/)

每条样本包含 `task` 字段区分两个子任务：

```json
// 领域分类 (domain)
{"text": "Title: Neural Machine Translation...
Abstract: We propose...", "label": "NLP", "task": "domain", "source": "acl_2017", "split": "train"}

// 质量分类 (quality)
{"text": "Title: Weak baseline...
Abstract: Our method...", "label": "reject", "task": "quality", "source": "iclr_2017", "split": "train"}
```

| 字段 | 说明 |
|:---|:---|
| `text` | 论文标题 + 摘要（换行符已清洗为空格） |
| `label` | `domain` 任务: `NLP`/`ML`/`AI`；`quality` 任务: `accept`/`reject` |
| `task` | `domain` 或 `quality` |
| `source` | 来源 venue |
| `split` | 原始划分: train/dev/test |

### Venue → Domain 映射

| Venue | Domain |
|:---|:---|
| `acl_2017`, `conll_2016`, `conll_2017`, `arxiv.cs.cl` | **NLP** |
| `iclr_2017`, `nips_2013-2017`, `arxiv.cs.lg` | **ML** |
| `arxiv.cs.ai` | **AI** |

### 使用建议
- 两个任务可**共享 SciBERT 编码器**，接两个独立分类头（多任务学习）
- `quality` 任务注意类别不平衡（accept 通常占 70-80%），建议加 `class_weight`
- 与 arXiv 合并时，PeerRead 的 domain 数据约 2,000 篇，arXiv 采样后约 2 万篇，比例约 1:10

---

## 2. 摘要/生成数据 (summarization/)

论文 → 审稿意见的生成对：

```json
{
  "source": "Title: Neural Machine Translation...
Abstract: We propose...",
  "target": "comments: well written
strengths: novel architecture
weaknesses: missing ablation study",
  "type": "review",
  "venue": "iclr_2017",
  "paper_id": "abc123"
}
```

| 字段 | 说明 |
|:---|:---|
| `source` | 输入：论文标题 + 摘要 |
| `target` | 输出：审稿人意见（多字段拼接） |
| `type` | 固定为 `review` |
| `venue` | 来源会议 |
| `paper_id` | 论文 ID |

### target 拼接字段
按优先级从 reviews 中提取：`comments` → `review` → `summary` → `strengths` → `weaknesses` → `questions` → `suggestion`

### 使用建议
- 训练模型自动生成审稿意见初稿
- 可与 SciTLDR 合并，扩展为"论文摘要 + 审稿意见"的多任务生成
- BART-base 建议 `max_source_length=512`、`max_target_length=256`

---

## 3. 检索数据 (retrieval/)

### 语料库 (corpus.jsonl)
```json
{"id": "acl_2017_abc123", "text": "Title:...
Abstract:...", "domain": "NLP", "venue": "acl_2017", "split": "train"}
```

### 对比学习训练对 (*_pairs.jsonl)
```json
{
  "anchor_id": "acl_2017_abc123",
  "anchor_text": "论文A摘要...",
  "positive_id": "acl_2017_def456",
  "positive_text": "同领域论文B摘要...",
  "negative_id": "iclr_2017_xyz789",
  "negative_text": "不同领域论文C摘要...",
  "domain": "NLP"
}
```

- **正样本**：同 venue / 同 domain 的论文
- **负样本**：不同 domain 随机采样
- 用于 SPECTER 等语义模型的**对比微调**（拉近正样本、推远负样本）

### 使用建议
- `corpus.jsonl` 可直接用 SPECTER 编码，构建 FAISS 索引
- `train_pairs.jsonl` 用于微调 SPECTER，让同领域论文向量更接近

---

## 规模

| 模块 | train | dev | test | 备注 |
|:---|:---|:---|:---|:---|
| 分类 (domain) | ~2,000 | ~300 | ~300 | 4 类: NLP/ML/AI |
| 分类 (quality) | ~10,000 | ~800 | ~800 | 2 类: accept/reject |
| 生成 (review) | ~8,000 | ~600 | ~600 | 审稿意见生成 |
| 检索语料 | ~15,000 | - | - | PeerRead 全部论文 |
| 检索对 | ~12,000 | ~700 | ~700 | 对比学习 |

---

## 注意事项

1. **reviews 嵌套在子文件夹**：原始 `reviews/` 文件夹下有多层子目录，脚本已递归遍历
2. **parsed_pdfs 嵌套在子文件夹**：论文解析结果在 `parsed_pdfs/` 下多层分布，脚本已递归读取
3. **venue 命名不统一**：不同会议的 JSON 命名略有差异，脚本已做兼容处理
4. **与 arXiv 合并**：PeerRead 的 domain 数据量较小，建议与 arXiv 采样数据合并后训练
