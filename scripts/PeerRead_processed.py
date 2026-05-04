#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PeerRead 预处理脚本 v2 — 兼容深层目录结构
支持: train/dev/test 下有 parsed_pdfs/、reviews/ 等多层子文件夹
输出: ./processed_data/
"""

import json
import random
from pathlib import Path
from collections import Counter, defaultdict

# ========== 配置 ==========
RAW_DIR = Path(__file__).resolve().parent.parent / "raw_data" / "PeerRead_data"
OUT_DIR = Path(__file__).resolve().parent.parent / "processed_data" / "PeerRead_processed_data"
SEED = 42
random.seed(SEED)

# venue -> 领域映射
VENUE2DOMAIN = {
    "acl_2017": "NLP",
    "conll_2016": "NLP",
    "conll_2017": "NLP",
    "arxiv.cs.ai_2007-2017": "AI",
    "arxiv.cs.cl_2007-2017": "NLP",
    "arxiv.cs.lg_2007-2017": "ML",
    "iclr_2017": "ML",
    "nips_2013-2017": "ML",
    "nips_2017": "ML",
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


def load_json_file(path: Path):
    """安全加载单个 JSON 文件"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]
    except Exception as e:
        print(f"    [WARN] 读取失败 {path}: {e}")
        return []


def clean_text(text):
    """清洗文本"""
    if not text:
        return ""
    return str(text).replace("\n", " ").replace("\r", " ").strip()


# ========== 模块 1: 分类数据 ==========
def build_classification():
    print("\n" + "=" * 60)
    print("[模块1] 构建分类数据集 (domain + quality)")
    print("=" * 60)

    save_dir = ensure_dir(OUT_DIR / "classification")
    train_data, dev_data, test_data = [], [], []

    if not RAW_DIR.exists():
        print(f"[ERROR] 找不到 {RAW_DIR}")
        return

    for venue_dir in sorted(RAW_DIR.iterdir()):
        if not venue_dir.is_dir():
            continue

        venue_name = venue_dir.name
        domain = VENUE2DOMAIN.get(venue_name, "OTHER")
        print(f"\n[PROCESS] Venue: {venue_name} -> Domain: {domain}")

        for split_name in ["train", "dev", "test"]:
            split_dir = venue_dir / split_name
            if not split_dir.exists():
                continue

            # 找论文 JSON：parsed_pdfs/ + reviews/ + reviews_raw/ + 直接目录
            direct = list(split_dir.glob(f"{venue_name}_{split_name}.json"))
            if not direct:
                direct = [
                    p
                    for p in split_dir.glob("*.json")
                    if "review" not in p.name.lower()
                ]
            pdf_jsons = (
                list((split_dir / "parsed_pdfs").rglob("*.json"))
                if (split_dir / "parsed_pdfs").exists()
                else []
            )
            review_jsons = (
                list((split_dir / "reviews").rglob("*.json"))
                if (split_dir / "reviews").exists()
                else []
            )
            rv_raw_jsons = (
                list((split_dir / "reviews_raw").rglob("*.json"))
                if (split_dir / "reviews_raw").exists()
                else []
            )

            all_jsons = direct + pdf_jsons + review_jsons + rv_raw_jsons
            if not all_jsons:
                print(f"  [SKIP] {split_dir} 未找到论文 JSON")
                continue

            # 加载所有 JSON 并去重 (同一篇论文可能同时出现在 parsed_pdfs 和 reviews 中)
            papers_raw = []
            for json_path in all_jsons:
                for paper in load_json_file(json_path):
                    # 把文件名信息注入到 paper 中，用于 id 推断
                    paper["_filename"] = json_path.stem
                    papers_raw.append(paper)

            # 用 paper_id 去重，优先保留有 accepted 字段的版本
            seen = {}
            for p in papers_raw:
                if "metadata" in p:
                    pid = p.get("name", p["_filename"]).replace(".pdf.json", "").replace(".pdf", "")
                else:
                    pid = str(p.get("id", p.get("paperId", "")))
                if not pid or pid == "None":
                    pid = p["_filename"]
                if not pid or pid == "None":
                    continue
                if pid not in seen or p.get("accepted") is not None or p.get("decision") is not None:
                    seen[pid] = p
            papers = list(seen.values())

            print(f"  [LOAD] {split_name}: {len(papers_raw)} 篇原始 -> {len(papers)} 篇去重后")

            target_list = {"train": train_data, "dev": dev_data, "test": test_data}.get(
                split_name, train_data
            )

            for p in papers:
                # 支持两种数据格式:
                #   parsed_pdfs: {name, metadata: {title, abstractText, ...}}
                #   reviews/reviews_raw: {id, title, abstract, reviews, accepted?, ...}
                if "metadata" in p:
                    md = p["metadata"]
                    title = clean_text(md.get("title", ""))
                    abstract = clean_text(md.get("abstractText", ""))
                else:
                    title = clean_text(p.get("title", ""))
                    abstract = clean_text(p.get("abstract", p.get("paperAbstract", "")))
                text = f"Title: {title}\nAbstract: {abstract}".strip()
                if len(text) < 20:
                    continue

                # domain
                target_list.append(
                    {
                        "text": text,
                        "label": domain,
                        "task": "domain",
                        "source": venue_name,
                        "split": split_name,
                    }
                )

                # quality
                decision = p.get("accepted", p.get("decision", None))
                label_quality = None
                if isinstance(decision, bool):
                    label_quality = "accept" if decision else "reject"
                elif isinstance(decision, str):
                    d = decision.lower()
                    if d in ["accept", "accepted", "true", "yes", "probably accept"]:
                        label_quality = "accept"
                    elif d in [
                        "reject",
                        "rejected",
                        "false",
                        "no",
                        "probably reject",
                        "borderline",
                    ]:
                        label_quality = "reject"
                elif isinstance(decision, int):
                    label_quality = "accept" if decision == 1 else "reject"

                if label_quality:
                    target_list.append(
                        {
                            "text": text,
                            "label": label_quality,
                            "task": "quality",
                            "source": venue_name,
                            "split": split_name,
                        }
                    )

    save_jsonl(train_data, save_dir / "train.jsonl")
    save_jsonl(dev_data, save_dir / "dev.jsonl")
    save_jsonl(test_data, save_dir / "test.jsonl")

    print("\n[STAT] 分类数据统计:")
    for name, data in [("train", train_data), ("dev", dev_data), ("test", test_data)]:
        c_domain = Counter([d["label"] for d in data if d["task"] == "domain"])
        c_quality = Counter([d["label"] for d in data if d["task"] == "quality"])
        print(f"  {name}: domain={dict(c_domain)}, quality={dict(c_quality)}")


# ========== 模块 2: 摘要/生成数据 ==========
def build_summarization():
    print("\n" + "=" * 60)
    print("[模块2] 构建摘要/生成数据集 (论文 -> 审稿意见)")
    print("=" * 60)

    save_dir = ensure_dir(OUT_DIR / "summarization")
    train_data, dev_data, test_data = [], [], []

    for venue_dir in sorted(RAW_DIR.iterdir()):
        if not venue_dir.is_dir():
            continue
        venue_name = venue_dir.name

        for split_name in ["train", "dev", "test"]:
            split_dir = venue_dir / split_name
            if not split_dir.exists():
                continue

            # 读取 reviews/ 文件夹下的所有 JSON
            reviews_dir = split_dir / "reviews"
            reviews_list = []
            if reviews_dir.exists():
                for rf in reviews_dir.rglob("*.json"):
                    reviews_list.extend(load_json_file(rf))
            else:
                fallback = split_dir / "reviews.json"
                if fallback.exists():
                    reviews_list.extend(load_json_file(fallback))

            if not reviews_list:
                continue

            # 读取论文元数据 (from parsed_pdfs + reviews)
            papers_map = {}
            direct = list(split_dir.glob(f"{venue_name}_{split_name}.json"))
            if not direct:
                direct = [
                    p
                    for p in split_dir.glob("*.json")
                    if "review" not in p.name.lower()
                ]
            pdf_jsons = (
                list((split_dir / "parsed_pdfs").rglob("*.json"))
                if (split_dir / "parsed_pdfs").exists()
                else []
            )
            rev_meta_jsons = (
                list((split_dir / "reviews").rglob("*.json"))
                if (split_dir / "reviews").exists()
                else []
            )

            for p in direct + pdf_jsons + rev_meta_jsons:
                for paper in load_json_file(p):
                    # Handle nested metadata from parsed_pdfs
                    if "metadata" in paper:
                        pid = paper.get("name", "").replace(".pdf.json", "").replace(".pdf", "")
                    else:
                        pid = paper.get("id", paper.get("paperId", ""))
                    if pid and str(pid) != "None":
                        papers_map[str(pid)] = paper

            count = 0
            target_list = {"train": train_data, "dev": dev_data, "test": test_data}.get(
                split_name, train_data
            )

            for rev_item in reviews_list:
                if not isinstance(rev_item, dict):
                    continue

                paper_id = rev_item.get("id", rev_item.get("paperId", ""))
                reviews = rev_item.get("reviews", [])
                if not isinstance(reviews, list):
                    reviews = [reviews]

                paper = papers_map.get(str(paper_id), {})
                if "metadata" in paper:
                    md = paper["metadata"]
                    title = clean_text(md.get("title", ""))
                    abstract = clean_text(md.get("abstractText", ""))
                else:
                    title = clean_text(paper.get("title", ""))
                    abstract = clean_text(
                        paper.get("abstract", paper.get("paperAbstract", ""))
                    )
                source_text = f"Title: {title}\nAbstract: {abstract}".strip()

                for rev in reviews:
                    if not isinstance(rev, dict):
                        continue

                    parts = []
                    for key in [
                        "comments",
                        "review",
                        "summary",
                        "strengths",
                        "weaknesses",
                        "questions",
                        "suggestion",
                    ]:
                        val = rev.get(key, "")
                        if val and str(val).strip():
                            parts.append(f"{key}: {clean_text(val)}")

                    target_text = "\n".join(parts)
                    if len(target_text) < 30:
                        continue

                    target_list.append(
                        {
                            "source": source_text,
                            "target": target_text,
                            "type": "review",
                            "venue": venue_name,
                            "paper_id": paper_id,
                        }
                    )
                    count += 1

            if count > 0:
                print(f"  [OK] {venue_name}/{split_name}: {count} 条 review")

    save_jsonl(train_data, save_dir / "train.jsonl")
    save_jsonl(dev_data, save_dir / "dev.jsonl")
    save_jsonl(test_data, save_dir / "test.jsonl")

    print(
        f"\n[STAT] 生成总计: train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}"
    )


# ========== 模块 3: 检索语料 ==========
def build_retrieval():
    print("\n" + "=" * 60)
    print("[模块3] 构建检索语料库")
    print("=" * 60)

    save_dir = ensure_dir(OUT_DIR / "retrieval")
    corpus = []

    for venue_dir in sorted(RAW_DIR.iterdir()):
        if not venue_dir.is_dir():
            continue
        venue_name = venue_dir.name
        domain = VENUE2DOMAIN.get(venue_name, "OTHER")

        for split_name in ["train", "dev", "test"]:
            split_dir = venue_dir / split_name
            if not split_dir.exists():
                continue

            all_jsons = []
            direct = list(split_dir.glob(f"{venue_name}_{split_name}.json"))
            if not direct:
                direct = [
                    p
                    for p in split_dir.glob("*.json")
                    if "review" not in p.name.lower()
                ]
            pdf_jsons = (
                list((split_dir / "parsed_pdfs").rglob("*.json"))
                if (split_dir / "parsed_pdfs").exists()
                else []
            )
            review_jsons = (
                list((split_dir / "reviews").rglob("*.json"))
                if (split_dir / "reviews").exists()
                else []
            )
            all_jsons = direct + pdf_jsons + review_jsons

            for p in all_jsons:
                for paper in load_json_file(p):
                    if "metadata" in paper:
                        md = paper["metadata"]
                        pid = paper.get("name", "").replace(".pdf.json", "").replace(".pdf", "")
                        title = clean_text(md.get("title", ""))
                        abstract = clean_text(md.get("abstractText", ""))
                    else:
                        pid = paper.get("id", paper.get("paperId", ""))
                        title = clean_text(paper.get("title", ""))
                        abstract = clean_text(
                            paper.get("abstract", paper.get("paperAbstract", ""))
                        )
                    text = f"Title: {title}\nAbstract: {abstract}".strip()
                    if len(text) < 50:
                        continue

                    corpus.append(
                        {
                            "id": f"{venue_name}_{pid}",
                            "text": text,
                            "domain": domain,
                            "venue": venue_name,
                            "split": split_name,
                        }
                    )

    with open(save_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  [SAVE] corpus.jsonl: {len(corpus)} 篇")

    # 对比学习训练对
    domain_groups = defaultdict(list)
    for item in corpus:
        domain_groups[item["domain"]].append(item)

    pairs = []
    for domain, papers in domain_groups.items():
        if len(papers) < 2:
            continue
        for anchor in papers:
            positives = [p for p in papers if p["id"] != anchor["id"]]
            if not positives:
                continue
            pos = random.choice(positives)
            neg_domain = random.choice([d for d in domain_groups.keys() if d != domain])
            neg = random.choice(domain_groups[neg_domain])

            pairs.append(
                {
                    "anchor_id": anchor["id"],
                    "anchor_text": anchor["text"],
                    "positive_id": pos["id"],
                    "positive_text": pos["text"],
                    "negative_id": neg["id"],
                    "negative_text": neg["text"],
                    "domain": domain,
                }
            )

    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * 0.9)
    n_dev = int(n * 0.05)

    save_jsonl(pairs[:n_train], save_dir / "train_pairs.jsonl")
    save_jsonl(pairs[n_train : n_train + n_dev], save_dir / "dev_pairs.jsonl")
    save_jsonl(pairs[n_train + n_dev :], save_dir / "test_pairs.jsonl")

    print(f"\n[STAT] 检索对: train={n_train}, dev={n_dev}, test={n - n_train - n_dev}")


# ========== 主入口 ==========
def main():
    print("[INIT] 原始数据目录:", RAW_DIR.resolve())
    print("[INIT] 输出目录:", OUT_DIR.resolve())
    ensure_dir(OUT_DIR)

    build_classification()
    build_summarization()
    build_retrieval()

    print("\n" + "=" * 60)
    print("[ALL DONE] PeerRead 处理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
