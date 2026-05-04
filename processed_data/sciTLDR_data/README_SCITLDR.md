# SciTLDR 数据说明

## 来源
[SciTLDR: Extreme Summarization of Scientific Documents](https://github.com/allenai/scitldr)  
官方仓库：`SciTLDR-Data/SciTLDR-A/`

## 原始文件

```
SciTLDR-A/
├── train.jsonl
├── dev.jsonl
└── test.jsonl
```

## 数据格式

```json
{
  "source": [
    "Due to the success of deep learning to solving a variety of challenging machine learning tasks...",
    "Particularly, the properties of critical points and the landscape around them are of importance...",
    "In this paper, we provide a necessary and sufficient characterization of the analytical forms..."
  ],
  "source_labels": [0, 0, 0, 0, 1, 0],
  "rouge_scores": [0.30, 0.37, 0.60, 0.57, 0.72, 0.15],
  "paper_id": "SysEexbRb",
  "target": ["We provide necessary and sufficient analytical forms for the critical points of the square loss functions for various neural networks..."],
  "title": "Critical Points of Linear Neural Networks: Analytical Forms and Landscape Properties"
}
```

## 字段说明

| 字段 | 类型 | 说明 | 训练时如何处理 |
|:---|:---|:---|:---|
| `source` | `list[str]` | 论文摘要的**句子数组** | `" ".join(source)` → 完整摘要文本 |
| `target` | `list[str]` | TLDR 摘要（通常只有 1 句） | `target[0]` → 目标生成文本 |
| `title` | `str` | 论文标题 | 建议拼接到 source 前：`title + ". " + source` |
| `source_labels` | `list[int]` | 抽取式摘要用的句子选中标记 | **生成式摘要可忽略** |
| `rouge_scores` | `list[float]` | 每句与 target 的 ROUGE 分数 | **可忽略** |
| `paper_id` | `str` | 论文内部 ID | **可忽略** |

## 转换为训练对

```python
import json

def load_scitldr(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            source = item["title"] + ". " + " ".join(item["source"])
            target = item["target"][0] if item["target"] else ""
            data.append({"source": source, "target": target, "paper_id": item.get("paper_id", "")})
    return data

# 使用示例
train = load_scitldr("SciTLDR-A/train.jsonl")
# train[0] = {
#   "source": "Critical Points of Linear Neural Networks... Due to the success...",
#   "target": "We provide necessary and sufficient analytical forms...",
#   "paper_id": "SysEexbRb"
# }
```

## 用途

- **训练论文摘要生成模型**：输入论文标题+摘要，输出一句话 TLDR
- 可作为 BART-base/large 或 T5-base/large 的 Seq2Seq 训练数据
- 可与 PeerRead reviews 合并，扩展为"论文摘要 + 审稿意见"的多任务生成

## 规模

| 数据集 | 数量 | 平均 source 长度 | 平均 target 长度 |
|:---|:---|:---|:---|
| train | ~3,000 对 | ~150 tokens | ~25 tokens |
| dev | ~500 对 | ~150 tokens | ~25 tokens |
| test | ~500 对 | ~150 tokens | ~25 tokens |

## 注意事项

- `source` 是**句子数组**，必须 `join` 后才能喂给 tokenizer
- `target` 是**单元素列表**，必须取 `[0]`
- 属于**极端摘要**（Extreme Summarization）：从几百词压缩到一句话，对生成模型挑战较大
- 建议 BART-base 的 `max_source_length=512`、`max_target_length=64`
