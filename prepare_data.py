"""Download and prepare PeerRead and SciERC datasets from HuggingFace.

All downloads go through HF_ENDPOINT (set https://hf-mirror.com for China).

Usage:
    uv run python prepare_data.py --all          # download both
    uv run python prepare_data.py --peerread     # classifier data only
    uv run python prepare_data.py --scierc       # NER data only
"""
import argparse
import json
import os
import sys

DATA_DIR = "data"


def prepare_peerread():
    """Download PeerRead from HF (allenai/peer_read), convert to classification format."""
    print("Downloading PeerRead from HuggingFace (allenai/peer_read, reviews)...")
    from datasets import load_dataset
    dataset = load_dataset("allenai/peer_read", "reviews", split="train", trust_remote_code=True)

    samples = []
    for paper in dataset:
        abstract = paper.get("abstract", "") or ""
        title = ""
        conference = str(paper.get("conference", "")).upper()

        if not abstract or len(abstract) < 50:
            continue

        domains = []
        if any(k in conference for k in ["ACL", "EMNLP", "NAACL", "NLP", "CONLL", "TACL"]):
            domains.append("NLP")
        if any(k in conference for k in ["CVPR", "ICCV", "ECCV"]):
            domains.append("CV")
        if any(k in conference for k in ["ICML", "NEURIPS", "NIPS"]):
            domains.append("ML")
        if any(k in conference for k in ["AAAI", "IJCAI", "ICLR"]):
            domains.append("AI")
        if not domains:
            domains = ["ML"]

        samples.append({
            "text": abstract.strip(),
            "domains": domains,
            "method_type": "Empirical",
        })

    if not samples:
        print("ERROR: No PeerRead samples extracted.")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "peerread_train.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"PeerRead: {len(samples)} samples -> {path}")


def prepare_scierc():
    """Download SciERC from HF (tner/scierc), convert to NER BIO format."""
    print("Downloading SciERC from HuggingFace (tner/scierc)...")
    from datasets import load_dataset

    all_samples = []
    for split in ["train", "validation", "test"]:
        try:
            dataset = load_dataset("tner/scierc", split=split, trust_remote_code=True)
        except Exception as e:
            print(f"  tner/scierc {split} failed: {e}")
            print("  Trying allenai/scierc...")
            try:
                dataset = load_dataset("allenai/scierc", split=split, trust_remote_code=True)
            except Exception:
                print(f"  Skipping split '{split}'")
                continue

        for doc in dataset:
            tokens = doc.get("tokens", doc.get("words", []))
            if not tokens:
                continue
            tags = doc.get("tags", doc.get("ner_tags", doc.get("labels", [])))
            if not tags:
                continue

            entities = _bio_to_entities(tags)
            all_samples.append({"tokens": tokens, "entities": entities})

    if not all_samples:
        print("ERROR: No SciERC samples. Trying alternative NER dataset...")
        all_samples = _fallback_ner_samples()

    n = len(all_samples)
    train_n = int(n * 0.8)

    os.makedirs(DATA_DIR, exist_ok=True)

    def save(name, data):
        path = os.path.join(DATA_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    p1 = save("scierc_train.json", all_samples[:train_n])
    p2 = save("scierc_val.json", all_samples[train_n:int(n * 0.9)])
    p3 = save("scierc_test.json", all_samples[int(n * 0.9):])
    print(f"SciERC: {n} sentences -> {p1} / {p2} / {p3}")


def _bio_to_entities(tags: list) -> list:
    """Convert BIO/IOB2 tag sequence to entity list [{type, start, end}]."""
    entities = []
    i = 0
    while i < len(tags):
        tag = str(tags[i])
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < len(tags) and str(tags[j]) == f"I-{etype}":
                j += 1
            entities.append({"type": etype, "start": i, "end": j})
            i = j
        else:
            i += 1
    return entities


def _fallback_ner_samples() -> list:
    print("WARNING: Using synthetic fallback NER data (for demonstration only)!")
    return [
        {
            "tokens": ["We", "use", "BERT", "on", "SQuAD", "and", "report", "F1", "."],
            "entities": [
                {"text": "BERT", "type": "MODEL", "start": 2, "end": 3},
                {"text": "SQuAD", "type": "DATASET", "start": 4, "end": 5},
                {"text": "F1", "type": "METRIC", "start": 7, "end": 8},
            ]
        }
        for _ in range(500)
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare training data from HuggingFace")
    p.add_argument("--all", action="store_true")
    p.add_argument("--peerread", action="store_true")
    p.add_argument("--scierc", action="store_true")
    args = p.parse_args()

    do_peerread = args.all or args.peerread
    do_scierc = args.all or args.scierc

    if not do_peerread and not do_scierc:
        p.print_help()
        sys.exit(0)

    if do_peerread:
        prepare_peerread()
    if do_scierc:
        prepare_scierc()
