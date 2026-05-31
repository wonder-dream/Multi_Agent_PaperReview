"""Evaluate NER model on SciERC test set."""
import torch
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score as seq_f1
from seqeval.metrics import classification_report as seq_report


def evaluate_ner(model, dataset, batch_size: int = 16, device: str = "cuda") -> dict:
    """Evaluate BiLSTM-CRF NER on a dataset.

    Returns dict with:
        entity_f1: entity-level F1 score
        report: per-entity-type classification report
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()

    all_pred_tags = []
    all_true_tags = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            tags = model.decode(input_ids, attention_mask)

            for i in range(len(tags)):
                true_seq = []
                pred_seq = []
                for j, (lid, pred) in enumerate(zip(labels[i], tags[i])):
                    if lid != -100:
                        true_seq.append(model.labels[lid])
                        pred_seq.append(pred)
                all_true_tags.append(true_seq)
                all_pred_tags.append(pred_seq)

    # Remove O tags for entity-level evaluation
    return {
        "entity_f1": float(seq_f1(all_true_tags, all_pred_tags)),
        "report": seq_report(all_true_tags, all_pred_tags),
    }
