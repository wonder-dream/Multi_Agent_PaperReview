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
    dataset_dict = load_dataset(
        "allenai/peer_read", "reviews",
        trust_remote_code=True,
        ignore_verifications=True,
    )
    dataset = dataset_dict["train"]

    samples = []
    for paper in dataset:
        abstract = paper.get("abstract", "") or ""
        title = ""
        conference = str(paper.get("conference", "")).upper()

        if not abstract or len(abstract) < 50:
            continue

        domains = []
        conf_upper = conference.upper()
        conf_lower = conference.lower()
        if any(k in conf_upper for k in ["ACL", "EMNLP", "NAACL", "HLT", "CONLL", "TACL"]):
            domains.append("NLP")
        if any(k in conf_upper for k in ["CVPR", "ICCV", "ECCV"]):
            domains.append("CV")
        if any(k in conf_upper for k in ["ICML", "NEURIPS", "NIPS"]):
            domains.append("ML")
        if any(k in conf_upper for k in ["AAAI", "IJCAI"]):
            domains.append("AI")
        if "ICLR" in conf_upper or "iclr" in conf_lower:
            domains.append("AI")
            domains.append("ML")
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
    """Download SciERC directly from UW server, convert to NER BIO format."""
    import tarfile
    import glob as _glob
    from urllib.request import urlretrieve

    print("Downloading SciERC from http://nlp.cs.washington.edu/sciIE/ ...")
    url = "http://nlp.cs.washington.edu/sciIE/data/sciERC_raw.tar.gz"
    tar_path = os.path.join(DATA_DIR, "scierc.tar.gz")

    os.makedirs(DATA_DIR, exist_ok=True)

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(block_num * block_size / total_size * 100, 100)
            downloaded = min(block_num * block_size, total_size)
            print(f"\r  {downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB ({pct:.0f}%)", end="")

    urlretrieve(url, tar_path, reporthook=_progress)
    print()
    print("  Downloaded, extracting...")

    extract_dir = os.path.join(DATA_DIR, "scierc_raw")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    # Parse BRAT format (.ann + .txt files), one sample per abstract
    txt_files = _glob.glob(os.path.join(extract_dir, "**", "*.txt"), recursive=True)
    print(f"  Found {len(txt_files)} text files in BRAT format")

    all_samples = []
    for txt_path in txt_files:
        ann_path = txt_path.replace(".txt", ".ann")
        if not os.path.exists(ann_path):
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()

        # Build token-to-character mapping
        tokens = raw_text.split()
        token_spans = []  # (start_char, end_char) per token
        pos = 0
        for tok in tokens:
            idx = raw_text.find(tok, pos)
            if idx == -1:
                token_spans.append((pos, pos + len(tok)))
                pos = pos + len(tok) + 1
            else:
                token_spans.append((idx, idx + len(tok)))
                pos = idx + len(tok)

        # Parse .ann entities
        entities = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("T"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                tag_parts = parts[1].split()
                if len(tag_parts) < 3:
                    continue
                etype = tag_parts[0]
                c_start = int(tag_parts[1])
                c_end = int(tag_parts[2])
                # Map character offsets to token indices
                tok_start = None
                tok_end = None
                for i, (ts, te) in enumerate(token_spans):
                    if tok_start is None and ts <= c_start < te:
                        tok_start = i
                    if ts <= c_end <= te:
                        tok_end = i + 1
                        break
                if tok_start is not None and tok_end is not None:
                    entities.append({
                        "text": parts[2],
                        "type": etype,
                        "start": tok_start,
                        "end": tok_end,
                    })

        if entities:
            all_samples.append({"tokens": tokens, "entities": entities})

    # Clean up temp files
    os.remove(tar_path)
    import shutil
    shutil.rmtree(extract_dir, ignore_errors=True)

    if not all_samples:
        print("ERROR: No SciERC samples extracted.")
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
