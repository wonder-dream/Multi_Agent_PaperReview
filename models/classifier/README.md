# 科研文本分类器 (Scientific Text Classifier)

基于 SciBERT 的科研论文多维度分类模块，支持领域分类、质量分类和多任务联合训练。

## 模块结构

```
classifier/
├── __init__.py              # 包初始化，导出主要类
├── scibert_classifier.py    # 核心模型定义
│   ├── SciBERTDomainClassifier      # 领域分类器
│   ├── SciBERTQualityClassifier     # 质量分类器
│   ├── SciBERTMultiTaskClassifier   # 多任务分类器
│   └── PaperClassifier              # Agent调用封装
├── dataset.py               # 数据集类
│   ├── DomainDataset        # 领域分类数据集
│   ├── QualityDataset       # 质量分类数据集
│   ├── MultiTaskDataset     # 多任务数据集
│   └── create_dataloaders() # DataLoader工厂
└── README.md                # 本文件
```

## 快速开始

### 1. 一键训练全部模型

```bash
python -m train.classifier.run_all \
    --processed_data_dir processed_data \
    --output_dir checkpoints \
    --models all \
    --batch_size 16 \
    --lr 2e-5 \
    --epochs 5
```

### 2. 单独训练领域分类器

```bash
python -m train.classifier.train_domain \
    --train_data processed_data/arxiv_PeerRead_merge_data/classification/merged_train.jsonl \
    --dev_data processed_data/arxiv_PeerRead_merge_data/classification/merged_dev.jsonl \
    --test_data processed_data/arxiv_PeerRead_merge_data/classification/merged_test.jsonl \
    --output_dir checkpoints/domain \
    --batch_size 16 \
    --lr 2e-5 \
    --epochs 5
```

### 3. 单独训练质量分类器

```bash
python -m train.classifier.train_quality \
    --train_data processed_data/arxiv_PeerRead_merge_data/classification/quality_train.jsonl \
    --dev_data processed_data/arxiv_PeerRead_merge_data/classification/quality_dev.jsonl \
    --test_data processed_data/arxiv_PeerRead_merge_data/classification/quality_test.jsonl \
    --output_dir checkpoints/quality \
    --batch_size 16 \
    --lr 2e-5 \
    --epochs 5 \
    --use_class_weights
```

### 4. 多任务联合训练

```bash
python -m train.classifier.train_multitask \
    --domain_train processed_data/arxiv_PeerRead_merge_data/classification/merged_train.jsonl \
    --domain_dev processed_data/arxiv_PeerRead_merge_data/classification/merged_dev.jsonl \
    --quality_train processed_data/arxiv_PeerRead_merge_data/classification/quality_train.jsonl \
    --quality_dev processed_data/arxiv_PeerRead_merge_data/classification/quality_dev.jsonl \
    --output_dir checkpoints/multitask \
    --domain_weight 1.0 \
    --quality_weight 1.0 \
    --batch_size 16 \
    --lr 2e-5 \
    --epochs 5
```

### 5. 评估模型

```bash
# 领域分类评估
python -m train.classifier.evaluate \
    --model_path checkpoints/domain/best_model.pt \
    --model_type domain \
    --test_data processed_data/arxiv_PeerRead_merge_data/classification/merged_test.jsonl \
    --output_dir results/domain

# 质量分类评估
python -m train.classifier.evaluate \
    --model_path checkpoints/quality/best_model.pt \
    --model_type quality \
    --test_data processed_data/arxiv_PeerRead_merge_data/classification/quality_test.jsonl \
    --output_dir results/quality
```

### 6. 推理/预测

```bash
# 单篇分类
python -m train.classifier.predict \
    --model_path checkpoints/domain/best_model.pt \
    --model_type domain \
    --text "Title: BERT: Pre-training... Abstract: We introduce..."

# 批量分类
python -m train.classifier.predict \
    --model_path checkpoints/multitask/best_model.pt \
    --model_type multitask \
    --input_file papers.json \
    --output_file predictions.json
```

### 7. Python API调用 (Agent层)

```python
from models.classifier import PaperClassifier

# 加载模型
classifier = PaperClassifier(
    model_path="checkpoints/multitask/best_model.pt",
    model_type="multitask"
)

# 单篇分类
result = classifier.classify("Title: xxx Abstract: xxx")
print(result)
# {
#     "domains": ["NLP"],
#     "method_type": "Empirical",
#     "quality_tier": "Acceptable",
#     "confidence": {"domain": 0.95, "quality": 0.88}
# }

# 批量分类
texts = ["Title: A...", "Title: B..."]
results = classifier.classify_batch(texts)

# Agent标准接口
from models.classifier.scibert_classifier import classify_paper
result = classify_paper("Title: xxx Abstract: xxx", classifier)
```

## 模型架构

### SciBERTDomainClassifier

```
Input: "Title: xxx Abstract: xxx" (max 512 tokens)
  |
  v
[SciBERT Encoder] (allenai/scibert_scivocab_uncased)
  |
  v
[CLS] Token (768-dim)
  |
  v
Dropout(0.1)
  |
  v
Linear(768 -> 4)
  |
  v
Output: [NLP, CV, ML, AI] 概率分布
```

### SciBERTQualityClassifier

```
Input: "Title: xxx Abstract: xxx" (max 512 tokens)
  |
  v
[SciBERT Encoder]
  |
  v
[CLS] Token (768-dim)
  |
  v
Dropout(0.1)
  |
  v
Linear(768 -> 2)
  |
  v
Output: [accept, reject] 概率分布 (支持class_weight)
```

### SciBERTMultiTaskClassifier

```
Input: "Title: xxx Abstract: xxx" (max 512 tokens)
  |
  v
[Shared SciBERT Encoder]
  |
  +---------> [CLS] ---------> Dropout ---------> Linear(768->4) --> Domain输出 [NLP, CV, ML, AI]
  |
  +---------> [CLS] ---------> Dropout ---------> Linear(768->2) --> Quality输出 [accept, reject]
```

## 评估指标

| 指标 | 说明 |
|------|------|
| Accuracy | 准确率 |
| Macro-F1 | 宏平均F1 (关注少数类) |
| Weighted-F1 | 加权平均F1 |
| Macro-Precision | 宏平均精确率 |
| Macro-Recall | 宏平均召回率 |
| Per-Class F1 | 每个类别的F1分数 |
| Confusion Matrix | 混淆矩阵 |

## 关键技术

### 类别不平衡处理 (Quality任务)

Quality任务中accept通常占70-80%，采用两种策略:

1. **类别权重**: `CrossEntropyLoss(weight=[1.0, 3.0])`，提高reject类权重
2. **自动计算**: `weight_i = total / (num_classes * count_i)`

### 多任务训练策略

1. **交替采样**: 每个step交替返回domain和quality样本
2. **过采样**: Quality样本较少时进行过采样，平衡两个任务的训练量
3. **损失加权**: `total_loss = w1 * domain_loss + w2 * quality_loss`
4. **标签掩码**: 使用-1标记无效标签，计算损失时忽略

### 早停机制

- 监控指标: Domain用Macro-F1，MultiTask用两个任务Macro-F1的平均值
- 耐心值: 默认3个epoch无改善则停止
- 最小改善量: 默认0.001

## 超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| batch_size | 16-32 | 根据显存调整 |
| lr | 2e-5 | SciBERT微调标准学习率 |
| epochs | 3-5 | 配合早停使用 |
| max_length | 512 | SciBERT最大长度 |
| dropout | 0.1 | 默认即可 |
| warmup_ratio | 0.1 | 10%步数预热 |
| patience | 3 | 早停耐心值 |

## 输入数据格式

### 领域分类 (JSONL)

```json
{"text": "Title: BERT... Abstract: We introduce...", "label": "NLP", "task": "domain", "source": "arxiv"}
```

### 质量分类 (JSONL)

```json
{"text": "Title: BERT... Abstract: We introduce...", "label": "accept", "task": "quality", "source": "iclr_2017"}
```

## 输出格式

```python
{
    "domains": ["NLP"],           # 领域列表 (多标签可扩展)
    "method_type": "Empirical",    # 方法类型 (预留)
    "quality_tier": "Acceptable",  # 质量等级映射
    "confidence": {                # 置信度
        "domain": 0.95,
        "quality": 0.88
    }
}
```
