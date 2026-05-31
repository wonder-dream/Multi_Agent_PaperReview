"""CLI: Compare pure LLM vs pure small model pipelines on a paper PDF.

Usage:
    uv run python compare.py --pdf paper.pdf --llm-api-key sk-xxx
    uv run python compare.py --pdf paper.pdf --deepseek-key sk-xxx --output report.json
"""
import argparse
import json
import os
import sys
import time

import torch
from dotenv import load_dotenv
load_dotenv()

CHECKPOINT_DIR = "checkpoints"
CLASSIFIER_CKPT = os.path.join(CHECKPOINT_DIR, "classifier", "best_model.pt")
NER_CKPT = os.path.join(CHECKPOINT_DIR, "ner", "best_model.pt")


def main():
    parser = argparse.ArgumentParser(description="LLM vs Small Model comparison")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--deepseek-key", default=os.environ.get("DEEPSEEK_API_KEY", ""),
                        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--output", default="comparison_report.json", help="Output JSON path")
    parser.add_argument("--device", default="cuda", help="Device for small models")
    args = parser.parse_args()

    # ---- Parse PDF ----
    print("=" * 60)
    print("Step 1: Parsing PDF...")
    from src.preprocessing.pdf_parser import parse_pdf
    text = parse_pdf(args.pdf)
    print(f"  Extracted {len(text):,} chars")

    # ---- Pipeline A: Pure LLM ----
    print("\n" + "=" * 60)
    print("Pipeline A: Pure LLM (DeepSeek V4 Pro)")
    llm_result = {"error": "No API key provided"}
    llm_time = 0

    if args.deepseek_key:
        from src.llm.client import LLMClient
        client = LLMClient(api_key=args.deepseek_key)

        t0 = time.time()
        classification = client.classify(text)
        entities = client.extract_entities(text)
        summary = client.summarize(text)
        checklist = client.check_manifest(text, entities, summary)
        llm_time = round(time.time() - t0, 1)

        llm_result = {
            "classification": classification,
            "entities": entities,
            "summary": summary,
            "checklist": checklist,
        }
        print(f"  Done in {llm_time}s")
        print(f"  Domains: {classification.get('domains', [])}")
        print(f"  Entities: {len(entities.get('entities', []))}")
        print(f"  Checklist items: {len(checklist)}")
    else:
        print("  Skipped (no API key)")

    # ---- Pipeline B: Pure Small Models ----
    print("\n" + "=" * 60)
    print("Pipeline B: Small Models (SciBERT)")
    from src.classifier.model import SciBERTMultiTaskClassifier
    from src.ner.model import BiLSTMCRFNER
    from src.summarizer.textrank import TextRankSummarizer
    from src.summarizer.checklist import ChecklistEngine

    # Load trained models (must match training architecture)
    classifier = SciBERTMultiTaskClassifier(pretrained=True).to(args.device)
    if os.path.exists(CLASSIFIER_CKPT):
        classifier.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=args.device))
        print(f"  Loaded classifier from {CLASSIFIER_CKPT}")
    else:
        print(f"  WARNING: {CLASSIFIER_CKPT} not found, using pretrained weights only")

    ner_model = BiLSTMCRFNER(pretrained=True).to(args.device)
    if os.path.exists(NER_CKPT):
        ner_model.load_state_dict(torch.load(NER_CKPT, map_location=args.device))
        print(f"  Loaded NER model from {NER_CKPT}")
    else:
        print(f"  WARNING: {NER_CKPT} not found, using pretrained weights only")

    summarizer = TextRankSummarizer()
    checklist_engine = ChecklistEngine()

    t0 = time.time()

    # Classify
    clf_result = classifier.predict_text(text[:2000])

    # NER with SciBERT tokenizer (must match model vocabulary)
    from src.preprocessing.sliding_window import chunk_text
    chunks = chunk_text(text, window_size=256, overlap=64)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    all_entities = []
    for chunk in chunks[:3]:
        tokens = tokenizer(chunk.text, max_length=256, truncation=True,
                          padding="max_length", return_tensors="pt")
        ner_out = ner_model.predict(
            tokens["input_ids"].to(args.device),
            tokens["attention_mask"].to(args.device),
            tokenizer=tokenizer,
        )
        all_entities.extend(ner_out.get("entities", []))

    # Summarize
    summary_text = summarizer.summarize(text, num_sentences=5)

    # Checklist
    checklist_items = checklist_engine.generate(all_entities, text)
    small_time = round(time.time() - t0, 1)

    small_result = {
        "classification": clf_result,
        "entities": all_entities,
        "summary": summary_text,
        "checklist": checklist_items,
    }
    print(f"  Done in {small_time}s")
    print(f"  Domains: {clf_result.get('domains', [])}")
    print(f"  Entities: {len(all_entities)}")
    print(f"  Checklist items: {len(checklist_items)}")

    # ---- Report ----
    report = {
        "paper": args.pdf,
        "text_length": len(text),
        "pipeline_a_llm": {"time_s": llm_time, **llm_result},
        "pipeline_b_small": {"time_s": small_time, **small_result},
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
