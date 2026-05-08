"""
信息抽取 Debug 训练脚本
支持 NER 和 RE 两种模式的快速验证

Usage:
    python -m train.extraction.debug_train --model_type ner --sample_size 100
    python -m train.extraction.debug_train --model_type re --sample_size 100
"""
import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.extraction.ner_model import SciBERTNERModel
from models.extraction.re_model import SciBERTRelationClassifier
from models.extraction.dataset import NERDataset, REDataset
from utils.classifier_utils import set_seed, get_device, print_model_info, format_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug训练 (信息抽取)")
    parser.add_argument("--model_type", type=str, required=True, choices=["ner", "re"])
    parser.add_argument("--processed_data_dir", type=str, default="processed_data")
    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--freeze_layers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--dev_sample_size", type=int, default=30)
    parser.add_argument("--random_sample", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="checkpoints/debug")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def sample_dataset(dataset, sample_size, seed=42):
    total = len(dataset)
    if sample_size >= total:
        return dataset
    np.random.seed(seed)
    indices = np.random.choice(total, sample_size, replace=False).tolist()
    from torch.utils.data import Subset
    return Subset(dataset, indices)


@torch.no_grad()
def evaluate_ner(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs["loss"].item()
        predictions = model.decode(input_ids, attention_mask)
        mask = attention_mask.bool().cpu()
        all_preds.extend([[p for p, m in zip(predictions[i], mask[i]) if m]
                          for i in range(len(predictions))])
        all_labels.extend([[labels[i][j].item() for j, m in enumerate(mask[i]) if m]
                           for i in range(len(labels))])

    from seqeval.metrics import f1_score, classification_report
    id2label = NERDataset.ID2LABEL
    pred_tags = [[id2label.get(t, "O") for t in seq] for seq in all_preds]
    true_tags = [[id2label.get(t, "O") for t in seq] for seq in all_labels]
    return {
        "loss": total_loss / len(dataloader),
        "f1": f1_score(true_tags, pred_tags),
        "report": classification_report(true_tags, pred_tags, output_dict=True)
    }


@torch.no_grad()
def evaluate_re(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs["loss"].item()
        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import f1_score, accuracy_score
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": float(accuracy_score(all_labels, np.array(all_preds))),
        "macro_f1": float(f1_score(all_labels, np.array(all_preds), average="macro", zero_division=0))
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    data_dir = os.path.join(args.processed_data_dir, "scierc_ner_re")

    if args.run_name is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        args.run_name = f"debug_{args.model_type}_{timestamp}"

    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Debug 训练 - {args.model_type.upper()}")
    logger.info(f"采样: {args.sample_size}条训练 / {args.dev_sample_size}条验证")

    logger.info("[1/4] 加载数据...")
    if args.model_type == "ner":
        train_dataset = NERDataset(os.path.join(data_dir, "ner_train.jsonl"), max_length=args.max_length)
        dev_dataset = NERDataset(os.path.join(data_dir, "ner_dev.jsonl"), max_length=args.max_length)
    else:
        train_dataset = REDataset(os.path.join(data_dir, "re_train.jsonl"), max_length=args.max_length)
        dev_dataset = REDataset(os.path.join(data_dir, "re_dev.jsonl"), max_length=args.max_length)

    train_dataset = sample_dataset(train_dataset, args.sample_size, args.seed)
    dev_dataset = sample_dataset(dev_dataset, args.dev_sample_size, args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"训练: {len(train_dataset)} | 验证: {len(dev_dataset)}")

    logger.info("[2/4] 创建模型...")
    if args.model_type == "ner":
        model = SciBERTNERModel(model_name=args.model_name, freeze_bert_layers=args.freeze_layers,
                                dropout_rate=args.dropout)
    else:
        model = SciBERTRelationClassifier(model_name=args.model_name, freeze_bert_layers=args.freeze_layers,
                                           dropout_rate=args.dropout)
    model.to(device)
    print_model_info(model)

    logger.info("[3/4] 开始训练...")
    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_loss += outputs["loss"].item()
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Step {batch_idx+1}/{len(train_loader)}, Loss: {outputs['loss'].item():.4f}")

        train_loss /= len(train_loader)
        eval_fn = evaluate_ner if args.model_type == "ner" else evaluate_re
        dev_metrics = eval_fn(model, dev_loader, device)

        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Dev Loss={dev_metrics['loss']:.4f}")
        if args.model_type == "ner":
            logger.info(f"  Dev F1: {dev_metrics['f1']:.4f}")
        else:
            logger.info(f"  Dev Acc: {dev_metrics['accuracy']:.4f}, Macro-F1: {dev_metrics['macro_f1']:.4f}")

    logger.info(f"总用时: {format_time(time.time() - start_time)}")

    logger.info("\n[4/4] 推理验证...")
    model.eval()
    if args.model_type == "ner":
        test_text = "BERT achieves state-of-the-art results on SQuAD with F1 score of 93.2 percent."
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        encoding = tokenizer(test_text, max_length=args.max_length, padding="max_length",
                             truncation=True, return_tensors="pt")
        preds = model.decode(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        tags = [NERDataset.ID2LABEL.get(p, "O") for p in preds[0]]
        logger.info(f"测试文本: '{test_text}'")
        entities = []
        for t, tag in zip(tokens, tags):
            if tag != "O" and not t.startswith("##"):
                entities.append(f"{t}({tag})")
        if entities:
            logger.info(f"  识别实体: {', '.join(entities)}")
        else:
            logger.info("  (未识别到实体)")
    else:
        sample = dev_dataset[0]
        outputs = model(input_ids=sample["input_ids"].unsqueeze(0).to(device),
                        attention_mask=sample["attention_mask"].unsqueeze(0).to(device))
        pred = torch.argmax(outputs["logits"], dim=-1).item()
        true = sample["labels"].item()
        logger.info(f"RE 推理: pred={REDataset.ID2RELATION[pred]}, true={REDataset.ID2RELATION[true]}")

    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(args.output_dir, "final_model.pt"))
    logger.info(f"模型已保存: {args.output_dir}")
    logger.info("一切正常!")


if __name__ == "__main__":
    main()
