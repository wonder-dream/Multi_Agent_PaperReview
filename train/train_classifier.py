"""Train SciBERT multi-task classifier on PeerRead data."""
import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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
    args = parser.parse_args()

    # Load data
    with open(args.data, "r", encoding="utf-8") as f:
        samples = json.load(f)

    dataset = PeerReadDataset(samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {args.device}")

    # Model
    print("Loading SciBERT...")
    model = SciBERTMultiTaskClassifier(pretrained=True).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    domain_criterion = nn.BCEWithLogitsLoss()
    method_criterion = nn.CrossEntropyLoss()

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
            loss_d = domain_criterion(out["domain_logits"], domain_labels)
            loss_m = method_criterion(out["method_logits"], method_label)
            loss = loss_d + loss_m

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, "best_model.pt")
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    main()
