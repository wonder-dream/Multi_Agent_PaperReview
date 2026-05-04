#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SciERC 独立预处理脚本
将 SciERC 原始 JSON 转换为:
  1) NER - BIO 格式 (token-level 标签序列)
  2) RE  - 关系分类样本 (含正负样本对)

"""

import json
import random
from pathlib import Path
from collections import Counter

# ========== 配置 ==========
RAW_DIR = Path(__file__).resolve().parent.parent / "raw_data" / "scierc_data"
OUT_DIR = Path(__file__).resolve().parent.parent / "processed_data" / "scierc_ner_re"
random.seed(42)

# SciERC 原始实体类型 -> 统一标签
ENTITY_MAP = {
    "Task": "TASK",
    "Method": "METHOD",
    "Metric": "METRIC",
    "Material": "DATASET",
    "OtherScientificTerm": "TERM",
    "Generic": "GENERIC",
}

# SciERC 原始关系类型 -> 统一标签
RELATION_MAP = {
    "USED-FOR": "USED-FOR",
    "FEATURE-OF": "FEATURE-OF",
    "HYPONYM-OF": "HYPONYM-OF",
    "PART-OF": "PART-OF",
    "COMPARE": "COMPARE-WITH",
    "CONJUNCTION": "CONJUNCTION",
    "EVALUATE-FOR": "EVALUATED-ON",
}


# ========== 工具函数 ==========
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_jsonl(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  [SAVE] {path.name}: {len(data)} 条")


# ========== NER: 转 BIO 格式 ==========
def build_ner():
    print("\n" + "=" * 60)
    print("[NER] 构建 BIO 格式数据集")
    print("=" * 60)

    for split in ["train", "dev", "test"]:
        raw_path = RAW_DIR / f"{split}.json"
        if not raw_path.exists():
            print(f"  [SKIP] {raw_path} 不存在")
            continue

        samples = []
        with open(raw_path, "r", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f if line.strip()]

        print(f"  [LOAD] SciERC {split}: {len(docs)} 篇文档")

        for doc in docs:
            sentences = doc["sentences"]  # list[list[str]]
            ner_annots = doc["ner"]  # list[list[[start,end,type]]]

            for sent_idx, (tokens, sent_ner) in enumerate(zip(sentences, ner_annots)):
                labels = ["O"] * len(tokens)
                entities = []

                for ent in sent_ner:
                    if len(ent) < 3:
                        continue
                    start, end, etype = ent[0], ent[1], ent[2]
                    mapped = ENTITY_MAP.get(etype, etype.upper())

                    # 合法性检查
                    if start < 0 or end >= len(tokens) or start > end:
                        continue
                    # 跳过与已标注实体重叠的实体 (避免 BIO 标签断裂)
                    if any(labels[i] != "O" for i in range(start, end + 1)):
                        continue

                    labels[start] = f"B-{mapped}"
                    for i in range(start + 1, end + 1):
                        labels[i] = f"I-{mapped}"

                    entities.append(
                        {
                            "text": " ".join(tokens[start : end + 1]),
                            "type": mapped,
                            "start": start,
                            "end": end,
                        }
                    )

                samples.append(
                    {
                        "doc_key": doc.get("doc_key", ""),
                        "sent_idx": sent_idx,
                        "tokens": tokens,
                        "labels": labels,
                        "entities": entities,
                    }
                )

        save_jsonl(samples, OUT_DIR / f"ner_{split}.jsonl")
        print(f"  [STAT] {split}: {len(samples)} 句")


# ========== RE: 关系分类样本 (含负样本) ==========
def build_re():
    print("\n" + "=" * 60)
    print("[RE] 构建关系分类数据集 (含正负样本)")
    print("=" * 60)

    for split in ["train", "dev", "test"]:
        raw_path = RAW_DIR / f"{split}.json"
        if not raw_path.exists():
            continue

        samples = []
        with open(raw_path, "r", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f if line.strip()]

        print(f"  [LOAD] SciERC {split}: {len(docs)} 篇文档")

        for doc in docs:
            sentences = doc["sentences"]
            ner_annots = doc["ner"]
            rel_annots = doc["relations"]

            for sent_idx, (tokens, sent_ner, sent_rel) in enumerate(
                zip(sentences, ner_annots, rel_annots)
            ):
                # 收集实体
                entities = []
                for ent in sent_ner:
                    if len(ent) < 3:
                        continue
                    start, end, etype = ent[0], ent[1], ent[2]
                    mapped = ENTITY_MAP.get(etype, etype.upper())
                    if start < 0 or end >= len(tokens) or start > end:
                        continue
                    entities.append(
                        {
                            "text": " ".join(tokens[start : end + 1]),
                            "type": mapped,
                            "start": start,
                            "end": end,
                        }
                    )

                if len(entities) < 2:
                    continue

                # 实体 span -> 实体 dict，用于快速查找
                entity_spans = {}
                for e in entities:
                    entity_spans[(e["start"], e["end"])] = e

                # 正样本: 从 sent_rel 读取
                pos_pairs = set()
                for rel in sent_rel:
                    if len(rel) < 5:
                        continue
                    s1, e1, s2, e2, rtype = rel[0], rel[1], rel[2], rel[3], rel[4]
                    mapped_rel = RELATION_MAP.get(rtype, rtype.upper())

                    ent1 = entity_spans.get((s1, e1))
                    ent2 = entity_spans.get((s2, e2))
                    if not ent1 or not ent2:
                        continue

                    pos_pairs.add((s1, e1, s2, e2))
                    pos_pairs.add(
                        (s2, e2, s1, e1)
                    )  # 关系通常是有向的，但这里标记为已存在

                    samples.append(
                        {
                            "doc_key": doc.get("doc_key", ""),
                            "sent_idx": sent_idx,
                            "sentence": " ".join(tokens),
                            "entity1_text": ent1["text"],
                            "entity1_type": ent1["type"],
                            "entity1_start": ent1["start"],
                            "entity1_end": ent1["end"],
                            "entity2_text": ent2["text"],
                            "entity2_type": ent2["type"],
                            "entity2_start": ent2["start"],
                            "entity2_end": ent2["end"],
                            "relation": mapped_rel,
                            "label": 1,
                        }
                    )

                # 负样本: 同句内无关系的实体对 (限制数量)
                max_neg = min(len(pos_pairs) // 2 + 1, 10) if pos_pairs else 5
                neg_count = 0

                for i, e1 in enumerate(entities):
                    for e2 in entities[i + 1 :]:
                        key = (e1["start"], e1["end"], e2["start"], e2["end"])
                        rkey = (e2["start"], e2["end"], e1["start"], e1["end"])
                        if key not in pos_pairs and rkey not in pos_pairs:
                            if neg_count >= max_neg:
                                break
                            neg_count += 1

                            samples.append(
                                {
                                    "doc_key": doc.get("doc_key", ""),
                                    "sent_idx": sent_idx,
                                    "sentence": " ".join(tokens),
                                    "entity1_text": e1["text"],
                                    "entity1_type": e1["type"],
                                    "entity1_start": e1["start"],
                                    "entity1_end": e1["end"],
                                    "entity2_text": e2["text"],
                                    "entity2_type": e2["type"],
                                    "entity2_start": e2["start"],
                                    "entity2_end": e2["end"],
                                    "relation": "NO-RELATION",
                                    "label": 0,
                                }
                            )
                    if neg_count >= max_neg:
                        break

        save_jsonl(samples, OUT_DIR / f"re_{split}.jsonl")
        pos = sum(1 for s in samples if s["label"] == 1)
        neg = sum(1 for s in samples if s["label"] == 0)
        print(f"  [STAT] {split}: 总计 {len(samples)} 对 (正{pos}/负{neg})")


# ========== 主入口 ==========
def main():
    print("[INIT] 原始数据目录:", RAW_DIR.resolve())
    print("[INIT] 输出目录:", OUT_DIR.resolve())
    ensure_dir(OUT_DIR)

    build_ner()
    build_re()

    print("\n" + "=" * 60)
    print("[ALL DONE] SciERC 处理完成!")
    print("=" * 60)
    print("""
输出结构:
├── processed_data/scierc_ner_re/
│   ├── ner_train.jsonl    (BIO 格式 NER)
│   ├── ner_dev.jsonl
│   ├── ner_test.jsonl
│   ├── re_train.jsonl     (关系分类: 正样本 + 负样本)
│   ├── re_dev.jsonl
│   └── re_test.jsonl
    """)


if __name__ == "__main__":
    main()
