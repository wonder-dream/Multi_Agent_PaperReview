# SciERC 清洗数据说明

## 输出文件

```
processed_data/scierc_ner_re/
├── ner_train.jsonl    # NER (BIO格式)
├── ner_dev.jsonl
├── ner_test.jsonl
├── re_train.jsonl     # 关系分类 (含正负样本)
├── re_dev.jsonl
└── re_test.jsonl
```

---

## 1. NER 数据 (BIO格式)

每句一个样本，token-level 标签序列：

```json
{
  "doc_key": "C88-1066",
  "sent_idx": 0,
  "tokens": ["This", "paper", "summarizes", "CCRs", "for", "NLP", "tasks"],
  "labels": ["O", "O", "O", "B-GENERIC", "O", "B-TASK", "I-TASK"],
  "entities": [
    {"text": "CCRs", "type": "GENERIC", "start": 3, "end": 3},
    {"text": "NLP tasks", "type": "TASK", "start": 5, "end": 6}
  ]
}
```

| 字段 | 说明 |
|:---|:---|
| `doc_key` | 原始文档 ID |
| `sent_idx` | 句子在文档中的序号 |
| `tokens` | 分词后的单词列表 |
| `labels` | BIO 标签序列，与 tokens **一一对应** |
| `entities` | 结构化实体列表（方便验证和调试） |

### 标签体系（6类）

SciERC 原始类型 → 统一映射：

| 原始类型 | 映射后 | 示例 |
|:---|:---|:---|
| `Task` | `TASK` | Machine Translation, NER |
| `Method` | `METHOD` | BERT, Fine-tuning |
| `Metric` | `METRIC` | F1, BLEU, Accuracy |
| `Material` | `DATASET` | SQuAD, ImageNet |
| `OtherScientificTerm` | `TERM` | CCRs, attention mechanism |
| `Generic` | `GENERIC` | model, approach |

### 使用建议
- 直接用于 `transformers` 的 `TokenClassification` 任务
- 输入构造：`tokenizer(tokens, is_split_into_words=True)`
- 注意 subword 对齐：BERT 会把一个词拆成多个 subword，需将 label 复制到第一个 subword，其余填 `-100`

---

## 2. RE 数据 (关系分类)

每个实体对一个样本，包含正负样本：

```json
// 正样本 (来自 SciERC 标注)
{
  "doc_key": "C88-1066",
  "sent_idx": 0,
  "sentence": "CCRs are Boolean conditions on the cooccurrence of categories ...",
  "entity1_text": "CCRs",
  "entity1_type": "GENERIC",
  "entity1_start": 0,
  "entity1_end": 0,
  "entity2_text": "categories",
  "entity2_type": "TERM",
  "entity2_start": 8,
  "entity2_end": 8,
  "relation": "USED-FOR",
  "label": 1
}

// 负样本 (同句无关系实体对，自动构造)
{
  "sentence": "...",
  "entity1_text": "CCRs",
  "entity1_type": "GENERIC",
  "entity1_start": 0,
  "entity1_end": 0,
  "entity2_text": "paper",
  "entity2_type": "GENERIC",
  "entity2_start": 1,
  "entity2_end": 1,
  "relation": "NO-RELATION",
  "label": 0
}
```

| 字段 | 说明 |
|:---|:---|
| `sentence` | 完整句子 |
| `entity1_text` / `entity2_text` | 两个实体的文本 |
| `entity1_type` / `entity2_type` | 实体类型 |
| `entity1_start` / `entity1_end` | 实体在句子中的 token 位置 |
| `relation` | 关系类型（正样本）或 `NO-RELATION`（负样本） |
| `label` | `1`=有关系, `0`=无关系 |

### 关系类型（7类）

| 原始关系 | 映射后 | 含义 |
|:---|:---|:---|
| `USED-FOR` | `USED-FOR` | 方法/模型用于解决某任务 |
| `FEATURE-OF` | `FEATURE-OF` | 某特征是另一实体的属性 |
| `HYPONYM-OF` | `HYPONYM-OF` | 下位词关系 |
| `PART-OF` | `PART-OF` | 部分-整体关系 |
| `COMPARE` | `COMPARE-WITH` | 与某方法/模型对比 |
| `CONJUNCTION` | `CONJUNCTION` | 并列关系 |
| `EVALUATE-FOR` | `EVALUATED-ON` | 在数据集/任务上评估 |

### 正负样本比例
- **正样本**：来自 SciERC 原始标注
- **负样本**：同句内未标注关系的实体对，随机采样
- 比例约 **1:1 ~ 1:2**，适合二分类或关系多分类

### BERT 输入构造示例
```
[CLS] CCRs are Boolean conditions on the cooccurrence of categories ... [SEP] entity1: CCRs [SEP] entity2: categories [SEP]
```

### 使用建议
- 可作为**句子级关系分类**（输入整句+实体标记，输出关系类型）
- 或作为**实体对级二分类**（输入两个实体上下文，输出 0/1）
- 负样本的 `NO-RELATION` 可单独作为一类，或合并到其他关系中做二分类

---

## 规模

| 数据集 | train | dev | test | 备注 |
|:---|:---|:---|:---|:---|
| NER | ~3,000 句 | ~1,000 句 | ~1,000 句 | BIO 格式 |
| RE | ~6,000 对 | ~2,000 对 | ~2,000 对 | 正样本 + 负样本 |

---

## 注意事项

1. **SciERC 原始格式是嵌套数组**：`sentences` 是 `list[list[str]]`，`ner` 和 `relations` 与句子一一对应，脚本已按句拆分
2. **实体位置是 token 级别**：不是字符级别，与 tokenizer 的 word_ids 对齐时需注意 subword 拆分
3. **负样本数量可控**：脚本中 `max_neg` 参数限制每句负样本数，避免极度不平衡
4. **与 ACE05 的区别**：SciERC 是科学文献领域，实体类型（Task/Method/Metric）与项目完全对齐；ACE05 是通用新闻领域，需付费获取
