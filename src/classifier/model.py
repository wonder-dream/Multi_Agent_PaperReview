"""Multi-task SciBERT classifier for domain and method type classification."""
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, AutoTokenizer


DOMAINS = ["NLP", "CV", "ML", "AI"]
METHODS = ["Empirical", "Theoretical", "Survey", "Benchmark"]


class SciBERTMultiTaskClassifier(nn.Module):
    """SciBERT with two heads: multi-label domain + single-label method type."""

    def __init__(
        self,
        num_domains: int = 4,
        num_methods: int = 4,
        pretrained: bool = True,
        model_name: str = "allenai/scibert_scivocab_uncased",
        hidden_size: int = 128,
        dropout: float = 0.3,
        freeze_layers: int = 0,
    ):
        super().__init__()
        self.num_domains = num_domains
        self.num_methods = num_methods

        if pretrained:
            self.bert = BertModel.from_pretrained(model_name)
        else:
            config = BertConfig(
                hidden_size=hidden_size,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=hidden_size * 4,
                vocab_size=31090,
            )
            self.bert = BertModel(config)

        if freeze_layers > 0:
            for p in self.bert.embeddings.parameters():
                p.requires_grad_(False)
            for i in range(min(freeze_layers, self.bert.config.num_hidden_layers)):
                for p in self.bert.encoder.layer[i].parameters():
                    p.requires_grad_(False)

        bert_dim = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.domain_head = nn.Linear(bert_dim, num_domains)     # multi-label
        self.method_head = nn.Linear(bert_dim, num_methods)     # single-label

        self._tokenizer = None
        self._model_name = model_name

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)

        return {
            "domain_logits": self.domain_head(pooled),
            "method_logits": self.method_head(pooled),
        }

    def predict(self, input_ids, attention_mask):
        """Return structured dict from tensor inputs."""
        self.eval()
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask)

        domain_probs = torch.sigmoid(out["domain_logits"]).squeeze(0)
        method_probs = torch.softmax(out["method_logits"], dim=-1).squeeze(0)

        domains = [DOMAINS[i] for i, p in enumerate(domain_probs) if p > 0.5]
        method_idx = method_probs.argmax().item()

        return {
            "domains": domains if domains else [DOMAINS[domain_probs.argmax().item()]],
            "method_type": METHODS[method_idx],
            "domain_probs": {d: round(float(domain_probs[i]), 4) for i, d in enumerate(DOMAINS)},
            "method_probs": {m: round(float(method_probs[i]), 4) for i, m in enumerate(METHODS)},
        }

    def predict_text(self, text: str) -> dict:
        """Convenience: classify a text string directly."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        tokens = self._tokenizer(
            text, max_length=512, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        return self.predict(
            tokens["input_ids"].to(device),
            tokens["attention_mask"].to(device),
        )
