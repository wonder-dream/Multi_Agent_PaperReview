"""Train BiLSTM-CRF NER model on SciERC data with early stopping."""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.ner.model import BiLSTMCRFNER
from src.ner.dataset import SciERCDataset, compute_sample_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="data/scierc_train.json")
    parser.add_argument("--val_data", default="data/scierc_val.json")
    parser.add_argument("--output", default="checkpoints/ner", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--freeze_layers", type=int, default=10,
                       help="Freeze bottom N SciBERT layers (default 10 of 12)")
    args = parser.parse_args()

    with open(args.train_data, "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(args.val_data, "r", encoding="utf-8") as f:
        val_samples = json.load(f)

    train_ds = SciERCDataset(train_samples)
    val_ds = SciERCDataset(val_samples)

    # Class-weighted sampling for METRIC tail class
    sample_weights = compute_sample_weights(train_samples, class_weight={"METRIC": 3.0})
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {args.device}")

    print("Loading SciBERT...")
    model = BiLSTMCRFNER(pretrained=True, freeze_layers=args.freeze_layers).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_val_f1 = 0.0
    patience_counter = 0

    def evaluate_on_val():
        from src.evaluation.eval_ner import evaluate_ner
        return evaluate_ner(model, val_ds, batch_size=args.batch_size, device=args.device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for batch in pbar:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)

            emissions, mask = model.forward(input_ids, attention_mask)
            loss = -model.crf(emissions, labels, mask=mask, reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        result = evaluate_on_val()
        val_f1 = result["entity_f1"]
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}  val_f1={val_f1:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            os.makedirs(args.output, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output, "best_model.pt"))
            print(f"  -> best model saved (val_f1={val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  -> no improvement (patience {patience_counter}/{args.early_stop_patience})")
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Done. Best val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
