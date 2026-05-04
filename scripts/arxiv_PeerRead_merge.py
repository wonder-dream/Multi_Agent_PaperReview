#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据合并脚本
将 arXiv 700MB 筛选数据（采样）与 PeerRead 分类数据合并
同时产出完整的检索语料库

输入:
  - raw_data/arxiv/arxiv_filtered_cs.jsonl  (700MB arXiv 筛选结果)
  - processed_data/classification/          (PeerRead 产出)

输出:
  - processed_data/classification/merged_*.jsonl      (合并后的分类训练集)
  - processed_data/classification/quality_*.jsonl     (PeerRead 质量分类，单独保留)
  - processed_data/retrieval/corpus_merged.jsonl      (完整检索语料: 50万+arxiv + peerread)
"""

import json
import random
from pathlib import Path
from collections import defaultdict, Counter

# ========== 配置 ==========
ARXIV_FILE = Path(__file__).resolve().parent.parent / "raw_data" / "arxiv_data" / "arxiv_filtered_cs.jsonl"
PEERREAD_DIR = Path(__file__).resolve().parent.parent / "processed_data" / "PeerRead_processed_data" / "classification"
OUT_DIR = Path(__file__).resolve().parent.parent / "processed_data" / "arxiv_PeerRead_merge_data"
SEED = 42
random.seed(SEED)

# arXiv 类别映射
CAT2DOMAIN = {
    "cs.CL": "NLP",
    "cs.CV": "CV",
    "cs.LG": "ML",
    "cs.AI": "AI",
}

# 每类最大采样数（避免淹没 PeerRead）
MAX_PER_DOMAIN = 5000


# ========== 工具函数 ==========
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_jsonl(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  [SAVE] {path.name}: {len(data)} 条")


def load_jsonl(path: Path):
    if not path.exists():
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def clean_text(text):
    return str(text).replace("\n", " ").replace("\r", " ").strip()


# ========== 1. 处理 arXiv: 采样 + 划分 ==========
def process_arxiv():
    print("\n" + "=" * 60)
    print("[1/3] 处理 arXiv 数据 (采样 + 划分)")
    print("=" * 60)

    if not ARXIV_FILE.exists():
        print(f"[ERROR] 找不到 {ARXIV_FILE}")
        return None, None

    # 按 domain + split 分组
    domain_split_groups = defaultdict(lambda: {"train": [], "dev": [], "test": []})
    total = 0

    with open(ARXIV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            if total % 100000 == 0:
                print(f"  [PROGRESS] 已读取 {total} 篇...")

            r = json.loads(line)
            categories = r.get("categories", [])
            domains = sorted(set(
                CAT2DOMAIN[cat] for cat in categories if cat in CAT2DOMAIN
            ))
            if not domains:
                continue

            year = r.get("year", 2000)
            tail = year % 10
            if tail <= 6:
                split = "train"
            elif tail <= 8:
                split = "dev"
            else:
                split = "test"

            title = clean_text(r.get("title", ""))
            abstract = clean_text(r.get("abstract", ""))
            text = f"Title: {title}\nAbstract: {abstract}".strip()
            if len(text) < 20:
                continue

            item = {
                "text": text,
                "label": ",".join(domains),  # 多标签
                "task": "domain",
                "source": "arxiv",
                "arxiv_id": r.get("id", ""),
                "year": year,
            }

            for d in domains:
                domain_split_groups[d][split].append(item)

    print(f"\n  [LOAD] arXiv 总计读取: {total} 篇")

    # 每类每 split 采样 (但总量控制: 每类所有 split 合计不超过 MAX_PER_DOMAIN)
    # 策略: 优先保证 train，其次 dev，最后 test
    sampled = {"train": [], "dev": [], "test": []}
    corpus = []

    for domain in ["NLP", "CV", "ML", "AI"]:
        all_items = []
        for split in ["train", "dev", "test"]:
            all_items.extend(domain_split_groups[domain][split])

        print(f"  [STAT] {domain}: 原始 {len(all_items)} 篇")

        # 随机打乱后采样
        random.shuffle(all_items)
        selected = all_items[:MAX_PER_DOMAIN]

        # 按原有 split 比例重新分配
        for item in selected:
            year = item["year"]
            tail = year % 10
            if tail <= 6:
                sampled["train"].append(item)
            elif tail <= 8:
                sampled["dev"].append(item)
            else:
                sampled["test"].append(item)

        # 检索语料: 全部保留 (不采样限制)
        for item in all_items:
            corpus.append(
                {
                    "id": f"arxiv_{item['arxiv_id']}",
                    "text": item["text"],
                    "domain": domain,
                    "source": "arxiv",
                    "venue": item["arxiv_id"].split("/")[0] if "/" in item["arxiv_id"] else domain,
                }
            )

    for split in ["train", "dev", "test"]:
        c = Counter([d["label"] for d in sampled[split]])
        print(f"  [SAMPLED] {split}: {len(sampled[split])} 条, 分布={dict(c)}")

    return sampled, corpus


# ========== 2. 处理 PeerRead: 分离 domain / quality ==========
def process_peerread():
    print("\n" + "=" * 60)
    print("[2/3] 处理 PeerRead 数据 (分离 domain / quality)")
    print("=" * 60)

    domain_data = {"train": [], "dev": [], "test": []}
    quality_data = {"train": [], "dev": [], "test": []}
    corpus = []

    for split in ["train", "dev", "test"]:
        path = PEERREAD_DIR / f"{split}.jsonl"
        items = load_jsonl(path)
        print(f"  [LOAD] PeerRead {split}: {len(items)} 条")

        for item in items:
            task = item.get("task", "")
            # 移除 split 字段避免冲突，保留原始来源
            clean_item = {
                "text": item["text"],
                "label": item["label"],
                "task": task,
                "source": item.get("source", "peerread"),
            }

            if task == "domain":
                domain_data[split].append(clean_item)
            elif task == "quality":
                quality_data[split].append(clean_item)

            # 检索语料: 所有 peerread 论文都加入 (去重文本)
            corpus.append(
                {
                    "id": f"peerread_{item.get('source', 'unknown')}_{hash(item['text']) % 10000000:07d}",
                    "text": item["text"],
                    "domain": item["label"] if task == "domain" else "UNKNOWN",
                    "source": "peerread",
                    "venue": item.get("source", "unknown"),
                }
            )

    print(
        f"\n  [STAT] PeerRead domain:  train={len(domain_data['train'])}, dev={len(domain_data['dev'])}, test={len(domain_data['test'])}"
    )
    print(
        f"  [STAT] PeerRead quality: train={len(quality_data['train'])}, dev={len(quality_data['dev'])}, test={len(quality_data['test'])}"
    )

    return domain_data, quality_data, corpus


# ========== 3. 合并与保存 ==========
def merge_and_save(arxiv_domain, arxiv_corpus, peer_domain, peer_quality, peer_corpus):
    print("\n" + "=" * 60)
    print("[3/3] 合并数据并保存")
    print("=" * 60)

    cls_dir = ensure_dir(OUT_DIR / "classification")
    ret_dir = ensure_dir(OUT_DIR / "retrieval")

    # --- 3.1 合并 domain 分类数据 ---
    for split in ["train", "dev", "test"]:
        merged = arxiv_domain[split] + peer_domain[split]
        random.shuffle(merged)
        save_jsonl(merged, cls_dir / f"merged_{split}.jsonl")

        c = Counter([d["label"] for d in merged])
        print(f"\n  [MERGED DOMAIN {split}]: {len(merged)} 条")
        print(
            f"    来源: arxiv={len(arxiv_domain[split])}, peerread={len(peer_domain[split])}"
        )
        print(f"    标签分布: {dict(c)}")

    # --- 3.2 保留 quality 分类数据 (仅 PeerRead) ---
    for split in ["train", "dev", "test"]:
        save_jsonl(peer_quality[split], cls_dir / f"quality_{split}.jsonl")

    q_train = len(peer_quality["train"])
    print(
        f"\n  [QUALITY] 仅 PeerRead: train={q_train}, dev={len(peer_quality['dev'])}, test={len(peer_quality['test'])}"
    )

    # --- 3.3 合并检索语料 (全部 arxiv + 全部 peerread) ---
    all_corpus = arxiv_corpus + peer_corpus
    random.shuffle(all_corpus)

    with open(ret_dir / "corpus_merged.jsonl", "w", encoding="utf-8") as f:
        for item in all_corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n  [SAVE] corpus_merged.jsonl: {len(all_corpus)} 篇")
    print(f"    来源: arxiv={len(arxiv_corpus)}, peerread={len(peer_corpus)}")

    # --- 3.4 构建检索对比学习对 (基于 corpus_merged) ---
    print("\n  [BUILD] 检索对比学习训练对...")
    domain_groups = defaultdict(list)
    for item in all_corpus:
        domain_groups[item["domain"]].append(item)

    pairs = []
    for domain, papers in domain_groups.items():
        if len(papers) < 2:
            continue
        # 每个 domain 最多生成 5000 对，避免过大
        count = 0
        max_pairs = min(len(papers) * 2, 5000)
        while count < max_pairs:
            anchor = random.choice(papers)
            positive = random.choice([p for p in papers if p["id"] != anchor["id"]])
            other_domains = [d for d in domain_groups.keys() if d != domain]
            if not other_domains:
                continue
            neg_domain = random.choice(other_domains)
            negative = random.choice(domain_groups[neg_domain])

            pairs.append(
                {
                    "anchor_id": anchor["id"],
                    "anchor_text": anchor["text"],
                    "positive_id": positive["id"],
                    "positive_text": positive["text"],
                    "negative_id": negative["id"],
                    "negative_text": negative["text"],
                    "domain": domain,
                }
            )
            count += 1

    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * 0.9)
    n_dev = int(n * 0.05)

    save_jsonl(pairs[:n_train], ret_dir / "merged_train_pairs.jsonl")
    save_jsonl(pairs[n_train : n_train + n_dev], ret_dir / "merged_dev_pairs.jsonl")
    save_jsonl(pairs[n_train + n_dev :], ret_dir / "merged_test_pairs.jsonl")

    print(
        f"\n  [STAT] 检索对: train={n_train}, dev={n_dev}, test={n - n_train - n_dev}"
    )


# ========== 主入口 ==========
def main():
    print("[INIT] arXiv 文件:", ARXIV_FILE.resolve())
    print("[INIT] PeerRead 目录:", PEERREAD_DIR.resolve())
    print("[INIT] 输出目录:", OUT_DIR.resolve())
    ensure_dir(OUT_DIR)

    arxiv_domain, arxiv_corpus = process_arxiv()
    if arxiv_domain is None:
        print("[ERROR] arXiv 处理失败，终止")
        return

    peer_domain, peer_quality, peer_corpus = process_peerread()
    merge_and_save(arxiv_domain, arxiv_corpus, peer_domain, peer_quality, peer_corpus)

    print("\n" + "=" * 60)
    print("[ALL DONE] 数据合并完成!")
    print("=" * 60)
    print("""
输出结构:
├── processed_data/arxiv_PeerRead_merge_data
│   ├── classification/
│   │   ├── merged_train.jsonl      (domain: arxiv采样 + peerread)
│   │   ├── merged_dev.jsonl
│   │   ├── merged_test.jsonl
│   │   ├── quality_train.jsonl     (quality: 仅 peerread)
│   │   ├── quality_dev.jsonl
│   │   └── quality_test.jsonl
│   └── retrieval/
│       ├── corpus_merged.jsonl       (全部arxiv + 全部peerread, 50万+篇)
│       ├── merged_train_pairs.jsonl  (对比学习三元组)
│       ├── merged_dev_pairs.jsonl
│       └── merged_test_pairs.jsonl

使用建议:
  1. 训练 domain 分类器: 用 merged_train/dev/test.jsonl
  2. 训练 quality 分类器: 用 quality_train/dev/test.jsonl
  3. 语义检索: 用 corpus_merged.jsonl 建索引，merged_train_pairs.jsonl 微调 SPECTER
    """)


if __name__ == "__main__":
    main()
