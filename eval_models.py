"""Evaluate trained classifier and NER models, save results to JSON."""
import json
import torch
from src.classifier.model import SciBERTMultiTaskClassifier
from src.classifier.dataset import PeerReadDataset
from src.evaluation.eval_classifier import evaluate_classifier
from src.ner.model import BiLSTMCRFNER
from src.ner.dataset import SciERCDataset
from src.evaluation.eval_ner import evaluate_ner

results = {}

# --- Classifier ---
# NOTE: PeerRead currently has no separate test split, so this evaluates on
# held-out samples from the same file used for training (first 500 samples).
# Domain F1 here will be optimistic compared to a true hold-out set.
print("Evaluating classifier...")
with open("data/peerread_train.json", encoding="utf-8") as f:
    samples = json.load(f)[:500]
clf_dataset = PeerReadDataset(samples)

clf_model = SciBERTMultiTaskClassifier(pretrained=True).to("cuda")
clf_model.load_state_dict(torch.load("checkpoints/classifier/best_model.pt", map_location="cuda", weights_only=True))
results["classifier"] = evaluate_classifier(clf_model, clf_dataset, device="cuda")
print(f"  Domain F1: {results['classifier']['domain_macro_f1']:.4f}")
print(f"  Method Acc: {results['classifier']['method_accuracy']:.4f}")

# --- NER ---
# Use scierc_test.json as the clean hold-out set.
# scierc_train.json is used for training; scierc_val.json for early stopping.
print("Evaluating NER on test set...")
with open("data/scierc_test.json", encoding="utf-8") as f:
    ner_test_samples = json.load(f)
ner_test_dataset = SciERCDataset(ner_test_samples)

ner_model = BiLSTMCRFNER(pretrained=True).to("cuda")
ner_model.load_state_dict(torch.load("checkpoints/ner/best_model.pt", map_location="cuda", weights_only=True))
results["ner_test"] = evaluate_ner(ner_model, ner_test_dataset, device="cuda")
print(f"  NER test F1: {results['ner_test']['entity_f1']:.4f}")

# --- Save ---
with open("outputs/evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nSaved to outputs/evaluation_results.json")
