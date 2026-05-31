"""Evaluate classifier on PeerRead test set."""
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


def evaluate_classifier(model, dataset, batch_size: int = 16, device: str = "cuda") -> dict:
    """Evaluate multi-task classifier on a dataset.

    Returns dict with:
        domain_f1: macro-F1 for multi-label domain classification
        method_accuracy: accuracy for single-label method type
        per_domain_f1: per-class F1 for each domain
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()

    all_domain_preds = []
    all_domain_labels = []
    all_method_preds = []
    all_method_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            domain_labels = batch["domain_labels"].cpu()
            method_label = batch["method_label"].cpu()

            out = model(input_ids, attention_mask)
            domain_preds = (torch.sigmoid(out["domain_logits"]) > 0.5).int().cpu()
            method_preds = out["method_logits"].argmax(dim=-1).cpu()

            all_domain_preds.append(domain_preds)
            all_domain_labels.append(domain_labels)
            all_method_preds.append(method_preds)
            all_method_labels.append(method_label)

    domain_preds = torch.cat(all_domain_preds).numpy()
    domain_labels = torch.cat(all_domain_labels).numpy()
    method_preds = torch.cat(all_method_preds).numpy()
    method_labels = torch.cat(all_method_labels).numpy()

    from src.classifier.model import DOMAINS

    per_domain_f1 = {}
    for i, name in enumerate(DOMAINS):
        if domain_labels[:, i].sum() > 0:
            per_domain_f1[name] = float(f1_score(domain_labels[:, i], domain_preds[:, i]))

    return {
        "domain_macro_f1": float(f1_score(domain_labels, domain_preds, average="macro")),
        "method_accuracy": float(accuracy_score(method_labels, method_preds)),
        "per_domain_f1": per_domain_f1,
    }
