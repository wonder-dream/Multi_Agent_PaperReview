"""SciBERT + BiLSTM + CRF for scientific Named Entity Recognition."""
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from .dataset import ENTITY_TYPES, LABEL2ID, ID2LABEL


def _build_labels():
    """BIO tag schema: B-{type}, I-{type}, O."""
    labels = ["O"]
    for t in ENTITY_TYPES:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    return labels


class BiLSTMCRFNER(nn.Module):
    """SciBERT encoder + BiLSTM + CRF decoder for NER."""

    def __init__(
        self,
        num_labels: int = None,
        pretrained: bool = True,
        model_name: str = "allenai/scibert_scivocab_uncased",
        hidden_size: int = 128,
        lstm_hidden: int = 384,
        dropout: float = 0.1,
    ):
        from torchcrf import CRF

        super().__init__()
        self.labels = _build_labels()
        self.num_labels = num_labels or len(self.labels)

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

        bert_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(bert_dim, lstm_hidden, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden * 2, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask):
        """Return emissions and mask for CRF."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(outputs.last_hidden_state)
        lstm_out, _ = self.bilstm(hidden)
        emissions = self.classifier(lstm_out)
        return emissions, attention_mask.bool()

    def decode(self, input_ids, attention_mask):
        """Decode to label sequences."""
        self.eval()
        with torch.no_grad():
            emissions, mask = self.forward(input_ids, attention_mask)
            tags = self.crf.decode(emissions, mask=mask)
        return [[self.labels[t] for t in seq] for seq in tags]

    def predict(self, input_ids, attention_mask, tokenizer=None):
        """Decode and extract entities, optionally decoding text with tokenizer."""
        tag_seqs = self.decode(input_ids, attention_mask)
        entities = []
        for i, seq in enumerate(tag_seqs):
            ids = input_ids[i] if tokenizer is not None else None
            entities.extend(_tags_to_entities(seq, ids, tokenizer))
        return {"entities": entities}


def _tags_to_entities(tags: list, input_ids=None, tokenizer=None) -> list:
    """Convert BIO tag sequence to entity list, decoding text if tokenizer given."""
    entities = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{etype}":
                j += 1
            text = ""
            if input_ids is not None and tokenizer is not None:
                text = tokenizer.decode(input_ids[i:j], skip_special_tokens=True)
            entities.append({"type": etype, "start": i, "end": j, "text": text})
            i = j
        else:
            i += 1
    return entities
