"""Train SciBERT multi-task classifier on PeerRead data with early stopping."""
import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.classifier.model import SciBERTMultiTaskClassifier
from src.classifier.dataset import PeerReadDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to PeerRead JSON file")
    parser.add_argument("--output", default="checkpoints/classifier", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--freeze_layers", type=int, default=10,
                       help="Freeze bottom N SciBERT layers (default 10 of 12)")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        samples = json.load(f)

    dataset = PeerReadDataset(samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {args.device}")

    print("Loading SciBERT...")
    model = SciBERTMultiTaskClassifier(
        pretrained=True, freeze_layers=args.freeze_layers,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    domain_criterion = nn.BCEWithLogitsLoss()
    method_criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    patience_counter = 0

    def evaluate_on_val():
        from src.evaluation.eval_classifier import evaluate_classifier
        return evaluate_classifier(model, val_ds, device=args.device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for batch in pbar:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            domain_labels = batch["domain_labels"].to(args.device)
            method_label = batch["method_label"].to(args.device)

            out = model(input_ids, attention_mask)
            loss = domain_criterion(out["domain_logits"], domain_labels) \
                 + method_criterion(out["method_logits"], method_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        result = evaluate_on_val()
        val_domain_f1 = result["domain_macro_f1"]
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}  domain_f1={val_domain_f1:.4f}  "
              f"method_acc={result['method_accuracy']:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(val_domain_f1)

        if val_domain_f1 > best_val_acc:
            best_val_acc = val_domain_f1
            patience_counter = 0
            os.makedirs(args.output, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output, "best_model.pt"))
            print(f"  -> best model saved (domain_f1={val_domain_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  -> no improvement (patience {patience_counter}/{args.early_stop_patience})")
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Done. Best domain F1: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
