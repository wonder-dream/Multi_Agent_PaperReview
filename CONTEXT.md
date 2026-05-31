# CONTEXT.md — 术语表与设计决策

## 核心概念

- **论文画像 (Paper Profile)**：论文使用了哪些数据集、模型、指标、方法的客观清单，不做价值判断。
- **完整性检查清单 (Completeness Checklist)**：基于论文内容逐项生成的检查条目（数据集多样性、Baseline 对比、消融实验、代码开源、伦理声明），每项状态为 ok/partial/missing/unchecked，不做 Accept/Reject 判断。
- **纯 LLM 方案 (LLM-Only Pipeline)**：DeepSeek V4 Pro 分 4 次结构化调用分别完成分类、NER、摘要、检查清单生成，LLM 输出即为最终结果。
- **纯小模型方案 (Small-Model Pipeline)**：SciBERT 微调分类器 + SciBERT-BiLSTM-CRF NER + TextRank 抽取式摘要，不使用 LLM 参与语义分析。
- **滑动窗口 (Sliding Window)**：解决小模型 512 token 限制的长文档处理策略，通过重叠分块 + 结果合并实现对全文的覆盖。具体参数通过超参实验确定（保守配置: 512+128 重叠+直接拼接 vs 激进配置: 512+256 重叠+段落边界感知+基于偏移合并），选 F1 更高的配置。
- **对比实验 (Comparison Experiment)**：两个方案在相同数据集上独立跑，用标准指标比较，不做混合交叉。

## 两个 Pipeline 结构

### Pipeline A: 纯 LLM

```
PDF 解析 → 全文文本
    │
    ├──→ LLM 调用 1: 分类 → {domains, method_type}
    ├──→ LLM 调用 2: NER → {entities: [{text, type, span}]}
    ├──→ LLM 调用 3: 摘要 → {structured_summary}
    └──→ LLM 调用 4: 检查清单 → {checklist_items}
```

每步独立 API 调用，输入均为全文，输出均为结构化 JSON。

### Pipeline B: 纯小模型

```
PDF 解析 → 全文文本
    │
    ├──→ 滑动窗口分块 → SciBERT 分类器 (每窗口预测 → 投票聚合) → {domains, method_type}
    ├──→ 滑动窗口分块 → BiLSTM-CRF NER (每窗口 B-I-O 标注 → 偏移合并去重) → {entities}
    └──→ TextRank + SciBERT 句子编码 → MMR 多样性惩罚 → {extractive_summary}
         └──→ 基于抽取实体 + 规则引擎 → {checklist_items}
```

所有模型为本地推理，不依赖外部 API。

## 评测方案

| 维度 | 数据集 | 指标 | Ground Truth |
|------|--------|------|-------------|
| 分类 | PeerRead test | Macro-F1, Accuracy | 论文原始标签 |
| NER | SciERC test | Span-F1 (entity-level) | 人工标注实体 |
| 摘要 | SciTLDR test | ROUGE-1/2/L, BERTScore | TLDR 参考摘要 |
| 检查清单 | — | 定性展示（两个方案输出并列） | 无 gold standard |

不引入 LLM-as-Judge，不做人工标注。检查清单部分仅做定性对比展示，不纳入定量评测。

## 模块裁剪

| 模块 | 决策 | 理由 |
|------|------|------|
| 分类器 (SciBERT) | 保留，需训练 | 有 PeerRead 标注数据，对比实验核心 |
| NER (BiLSTM-CRF) | 保留，需训练 | 有 SciERC 标注数据，对比实验核心 |
| 关系抽取 (RE) | 砍掉 | SciERC RE F1 仅 ~50%，训练复杂度高，不贡献核心对比 |
| 摘要 | 改为 TextRank 抽取式 | 零训练，BART 训练成本高且 SciTLDR 仅 5K 条数据 |
| 检索 (SPECTER2+FAISS) | 砍掉 | 两个方案共享同一检索，无对比价值 |
| Agent 层 | 砍掉 | 原实现为硬编码字符串，无实际推理 |
| 前端 | 砍掉 | 交付物为 CLI + 评测脚本 + 报告 |

## 团队分工（供参考）

| 姓名 | 负责 |
|------|------|
| 覃疆楠 | 小模型训练（分类 + NER）+ 评测脚本 |
| 李明远 | 数据处理（PeerRead/SciERC/SciTLDR 预处理）+ TextRank 摘要 + 检查清单规则引擎 |
| 许世典 | PDF 解析 + 滑动窗口分块器 + LLM 调用模块 + CLI 对比入口 |

## 项目结构（目标）

```
Multi_Agent_PaperReview/
├── src/
│   ├── classifier/          # SciBERT 分类器训练 + 推理
│   ├── ner/                 # BiLSTM-CRF NER 训练 + 推理
│   ├── summarizer/          # TextRank 抽取式摘要 + 检查清单规则引擎
│   ├── llm/                 # DeepSeek API 调用 + prompt 模板
│   ├── preprocessing/       # PDF 解析 + 滑动窗口分块
│   └── evaluation/          # 评测脚本 (分类/NER/摘要)
├── train/                   # 训练入口脚本
├── compare.py               # CLI 对比入口
├── CONTEXT.md
├── docs/adr/
└── data/                    # 预处理后的数据集
```
