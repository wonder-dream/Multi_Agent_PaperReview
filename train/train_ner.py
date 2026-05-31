"""Train BiLSTM-CRF NER model on SciERC data."""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.ner.model import BiLSTMCRFNER
from src.ner.dataset import SciERCDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to SciERC JSON file")
    parser.add_argument("--output", default="checkpoints/ner", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        samples = json.load(f)

    dataset = SciERCDataset(samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {args.device}")

    print("Loading SciBERT...")
    model = BiLSTMCRFNER(pretrained=True).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, "best_model.pt")
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    main()
