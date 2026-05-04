#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArXiv 筛选数据预处理脚本
将 extract_arxiv_windows.py 产出的 arxiv_filtered_cs.jsonl
转换为分类模型可直接使用的 train/dev/test 格式

输出: ./processed_data/classification/arxiv_*.jsonl
"""

import json
import random
from pathlib import Path
from collections import Counter

# ========== 配置 ==========
RAW_FILE = Path(__file__).resolve().parent.parent / "raw_data" / "arxiv_data" / "arxiv_filtered_cs.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent / "processed_data" / "classification_arxiv"
SEED = 42
random.seed(SEED)

# arXiv 类别 -> 统一领域标签
CAT2DOMAIN = {
    "cs.CL": "NLP",
    "cs.CV": "CV",
    "cs.LG": "ML",
    "cs.AI": "AI",
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


# ========== 主处理 ==========
def main():
    print("[INIT] 输入文件:", RAW_FILE.resolve())
    print("[INIT] 输出目录:", OUT_DIR.resolve())
    ensure_dir(OUT_DIR)

    if not RAW_FILE.exists():
        print(f"[ERROR] 找不到 {RAW_FILE}")
        print("[HINT] 请先运行 extract_arxiv_windows.py 提取数据")
        return

    # 加载
    records = []
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"[LOAD] 共 {len(records)} 篇论文")

    # 转换样本
    samples = []
    for r in records:
        title = r.get("title", "").strip()
        abstract = r.get("abstract", "").strip()
        text = f"Title: {title}\nAbstract: {abstract}".strip()

        if len(text) < 20:
            continue

        primary = r.get("primary_category", "")
        domain = CAT2DOMAIN.get(primary, "")
        if not domain:
            continue

        samples.append(
            {
                "text": text,
                "label": domain,
                "task": "domain",
                "source": "arxiv",
                "arxiv_id": r.get("id", ""),
                "year": r.get("year", 0),
                "categories": r.get("categories", []),
            }
        )

    print(f"[STAT] 有效样本: {len(samples)}")

    # 按年份划分: 尾数 0-6 train, 7-8 dev, 9 test
    train_data, dev_data, test_data = [], [], []
    year_counts = Counter()

    for s in samples:
        year = s["year"]
        year_counts[year] += 1
        tail = year % 10
        if tail <= 6:
            train_data.append(s)
        elif tail <= 8:
            dev_data.append(s)
        else:
            test_data.append(s)

    # 保存
    save_jsonl(train_data, OUT_DIR / "arxiv_train.jsonl")
    save_jsonl(dev_data, OUT_DIR / "arxiv_dev.jsonl")
    save_jsonl(test_data, OUT_DIR / "arxiv_test.jsonl")

    # 统计
    print("\n[STAT] 领域分布:")
    for name, data in [("train", train_data), ("dev", dev_data), ("test", test_data)]:
        c = Counter([d["label"] for d in data])
        print(f"  {name}: {len(data)} 条, 分布={dict(c)}")

    print("\n[STAT] 年份范围:", f"{min(year_counts)}-{max(year_counts)}")

    print("\n" + "=" * 60)
    print("[ALL DONE] ArXiv 分类数据准备完成!")
    print("=" * 60)
    print(f"""
输出文件:
├── processed_data/classification_arxiv/
│   ├── arxiv_train.jsonl    (约 {len(train_data)} 条)
│   ├── arxiv_dev.jsonl      (约 {len(dev_data)} 条)
│   └── arxiv_test.jsonl     (约 {len(test_data)} 条)

使用方式:
  1. 单独训练 ArXiv 领域分类器
  2. 与 PeerRead 的分类数据合并，做多任务/多源训练
    """)


if __name__ == "__main__":
    main()
