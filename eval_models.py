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
print("Evaluating classifier...")
with open("data/peerread_train.json", encoding="utf-8") as f:
    samples = json.load(f)[:500]
clf_dataset = PeerReadDataset(samples)

clf_model = SciBERTMultiTaskClassifier(pretrained=True).to("cuda")
clf_model.load_state_dict(torch.load("checkpoints/classifier/best_model.pt", map_location="cuda"))
results["classifier"] = evaluate_classifier(clf_model, clf_dataset, device="cuda")
print(f"  Domain F1: {results['classifier']['domain_macro_f1']:.4f}")
print(f"  Method Acc: {results['classifier']['method_accuracy']:.4f}")

# --- NER ---
print("Evaluating NER...")
for split_name, split_file in [("train", "scierc_train.json"), ("test", "scierc_test.json")]:
    with open(f"data/{split_file}", encoding="utf-8") as f:
        ner_samples = json.load(f)
    ner_dataset = SciERCDataset(ner_samples)

    ner_model = BiLSTMCRFNER(pretrained=True).to("cuda")
    ner_model.load_state_dict(torch.load("checkpoints/ner/best_model.pt", map_location="cuda"))
    results[f"ner_{split_name}"] = evaluate_ner(ner_model, ner_dataset, device="cuda")
    print(f"  NER {split_name} F1: {results[f'ner_{split_name}']['entity_f1']:.4f}")

# --- Save ---
with open("outputs/evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nSaved to evaluation_results.json")
