# 基于对比实验的科研文献分析系统

NLP 大作业 · 应用型选题

## 一句话概述

输入一篇学术论文 PDF，同时跑**纯 LLM**（DeepSeek V4 Pro）和**纯小模型**（SciBERT）两条 pipeline，对比两者的分类/实体识别/摘要生成性能，输出对比报告。

## 为什么这样做

原"Agent 编排 + 4 个 NLP 模块"架构存在三个致命问题：
1. 所有模型受 512 token 限制，实际只能分析摘要而非全文
2. Agent 层是硬编码字符串，没有真实的 LLM 推理决策
3. 多模块级联误差传播，端到端有效性能可能低于 50%

新方案将项目定位从"构建一个审稿系统"转为**"做一个对比实验"**：用标准 NLP 测试集（PeerRead / SciERC / SciTLDR）作为 ground truth，定量回答"LLM 是否已经好到可以替代专业小模型"。

详见 `docs/adr/0001-llm-vs-small-model-comparison.md`。

## 快速开始

### 1. 环境配置

```bash
# 安装 uv（如果没有）
pip install uv

# 同步依赖
uv sync
```

### 2. 设置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API key
```

### 3. 跑一轮对比

```bash
uv run python compare.py --pdf pdfs/your-paper.pdf
```

输出 `comparison_report.json`，包含两条 pipeline 的全部结果。

## 项目结构

```
├── src/
│   ├── preprocessing/       # PDF 解析 + 滑动窗口分块
│   ├── llm/                 # DeepSeek API（OpenAI 兼容协议）
│   ├── classifier/          # SciBERT 多任务分类器
│   ├── ner/                 # BiLSTM-CRF 命名实体识别
│   ├── summarizer/          # TextRank 摘要 + 检查清单规则引擎
│   └── evaluation/          # 评测：F1 / ROUGE / BERTScore
├── train/                   # 训练入口
├── tests/                   # 39 个单元测试
├── compare.py               # 主 CLI 入口
├── CONTEXT.md               # 术语表
├── docs/adr/                # 架构决策记录
└── pyproject.toml           # uv 依赖管理
```

## 两条 Pipeline 详解

### Pipeline A：纯 LLM（DeepSeek V4 Pro）

```
PDF → 全文文本 → 4 次独立 API 调用:
  1. classify(text)    → {domains, method_type}
  2. extract(text)     → {entities: [{text, type}]}
  3. summarize(text)   → {background, contributions, ...}
  4. checklist(text)   → [{category, status, detail}]
```

- 每次调用的 prompt 模板在 `src/llm/prompts.py`
- API 客户端在 `src/llm/client.py`
- 支持自动重试（最多 2 次）

### Pipeline B：纯小模型

```
PDF → 全文文本 → 滑动窗口分块 → 三条独立路径:
  1. SciBERT 分类器（每窗口预测 → 投票聚合）→ {domains, method_type}
  2. BiLSTM-CRF NER（每窗口标注 → 偏移合并）→ {entities}
  3. TextRank 抽取式摘要 + 规则引擎 → {summary, checklist}
```

- 分类器：`src/classifier/model.py` — SciBERT + Domain head + Method head
- NER：`src/ner/model.py` — SciBERT + BiLSTM(384×2) + CRF（11 类 BIO 标签）
- 摘要：`src/summarizer/textrank.py` — TF-IDF / SciBERT 句子编码 + PageRank + MMR
- 检查清单：`src/summarizer/checklist.py` — 6 类规则检查

## 模块接口速查

### PDF 解析
```python
from src.preprocessing.pdf_parser import parse_pdf
text = parse_pdf("paper.pdf")  # → str
```
MinerU CLI 优先（输出 markdown），PyMuPDF 降级。

### 滑动窗口
```python
from src.preprocessing.sliding_window import chunk_text
chunks = chunk_text(text, window_size=512, overlap=128)
# → [Chunk(text="...", start=0, end=512, index=0), ...]
```

### LLM 调用
```python
from src.llm.client import LLMClient
client = LLMClient(api_key="sk-xxx")
client.classify(text)          # → {domains, method_type}
client.extract_entities(text)  # → {entities, relations}
client.summarize(text)         # → {background, contributions, ...}
client.check_manifest(text, entities, summary)  # → [{category, status, detail}]
```

### 分类器
```python
from src.classifier.model import SciBERTMultiTaskClassifier
model = SciBERTMultiTaskClassifier(pretrained=True)
model.predict_text("This paper...")
# → {domains: ["NLP"], method_type: "Empirical", domain_probs: {...}, method_probs: {...}}
```

### NER
```python
from src.ner.model import BiLSTMCRFNER
model = BiLSTMCRFNER(pretrained=True)
model.predict(input_ids, attention_mask)
# → {entities: [{type: "MODEL", start: 0, end: 1, text: ""}]}
```

### 摘要与检查清单
```python
from src.summarizer.textrank import TextRankSummarizer
from src.summarizer.checklist import ChecklistEngine

s = TextRankSummarizer()
s.summarize(text, num_sentences=5)       # → List[str]
s.structured_summary(text)                # → {background, contributions, ...}

e = ChecklistEngine()
e.generate(entities, paper_text=text)     # → [{category, status, detail}]
```

## 数据准备

### PeerRead（分类训练）
从 https://github.com/allenai/PeerRead 下载，处理为 JSON 列表：
```json
[
  {"text": "标题+摘要", "domains": ["NLP"], "method_type": "Empirical"},
  ...
]
```

### SciERC（NER 训练）
从 https://github.com/dwadden/LOSTIN 下载，处理为 JSON 列表：
```json
[
  {
    "tokens": ["We", "use", "BERT", "on", "SQuAD"],
    "entities": [
      {"text": "BERT", "type": "MODEL", "start": 2, "end": 3},
      {"text": "SQuAD", "type": "DATASET", "start": 4, "end": 5}
    ]
  },
  ...
]
```

### SciTLDR（摘要评测）
从 https://github.com/neulab/SciTLDR 下载，处理为 JSON 列表：
```json
[
  {"text": "论文全文", "summary": "参考TLDR摘要"},
  ...
]
```

## 训练

### 训分类器
```bash
uv run python -m train.train_classifier \
    --data data/peerread_train.json \
    --output checkpoints/classifier \
    --epochs 5 --batch_size 16
```

### 训 NER
```bash
uv run python -m train.train_ner \
    --data data/scierc_train.json \
    --output checkpoints/ner \
    --epochs 10 --batch_size 16
```

5090 上这两个训练各约 1-2 小时（取决于数据量）。

## 评测

```python
# 分类评测
from src.evaluation.eval_classifier import evaluate_classifier
from src.classifier.dataset import PeerReadDataset

dataset = PeerReadDataset(test_samples)
results = evaluate_classifier(model, dataset)
# → {domain_macro_f1: 0.85, method_accuracy: 0.88, per_domain_f1: {...}}

# NER 评测
from src.evaluation.eval_ner import evaluate_ner
results = evaluate_ner(ner_model, scierc_dataset)
# → {entity_f1: 0.80, report: "..."}

# 摘要评测
from src.evaluation.eval_summarizer import evaluate_summarizer
results = evaluate_summarizer(summarizer, references, texts)
# → {rouge1: 0.42, rouge2: 0.18, rougeL: 0.38}
```

## 运行测试

```bash
uv run pytest -v              # 全部 39 个测试
uv run pytest tests/test_llm.py -v    # 只跑 LLM 模块
```

## 团队分工

| 姓名 | 负责模块 | 关键文件 |
|------|---------|---------|
| 覃疆楠（队长） | 小模型训练 + 评测 | `src/classifier/`, `src/ner/`, `train/`, `src/evaluation/` |
| 李明远 | 数据处理 + 摘要 | `src/summarizer/`, 数据预处理脚本 |
| 许世典 | PDF + LLM + CLI | `src/preprocessing/`, `src/llm/`, `compare.py` |

## 术语表

核心术语定义见 `CONTEXT.md`。关键概念：

- **论文画像**：论文用了哪些数据集/模型/指标/方法的客观清单
- **检查清单**：6 类结构化检查项（数据集多样性、Baseline、消融、代码、伦理、指标），每项为 ok/partial/missing/unchecked
- **纯 LLM 方案**：DeepSeek 分 4 次调用完成全部分析
- **纯小模型方案**：SciBERT 分类 + BiLSTM-CRF NER + TextRank 摘要，本地推理

## 参考文档

- `CONTEXT.md` — 完整术语表与设计决策
- `docs/adr/0001-llm-vs-small-model-comparison.md` — 架构决策记录
- `开题报告.md` — 原始开题报告（选题背景、创新点）
- `report.md` — 可行性评估报告（SOTA 对比、数据集分析）
- `INSPECTION_REPORT.md` — 原架构检验报告（记录了重做的原因）
