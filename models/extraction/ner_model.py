"""
NER模型: SciBERT + BiLSTM + CRF
"""
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, List


class SciBERTNERModel(nn.Module):
    """
    SciBERT + BiLSTM + CRF 命名实体识别模型

    架构: SciBERT(全序列输出) → BiLSTM → Linear → CRF
    """

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_labels: int = 13,
        lstm_hidden: int = 384,
        dropout_rate: float = 0.1,
        freeze_bert_layers: int = 0
    ):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size

        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.bilstm = nn.LSTM(hidden_size, lstm_hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _freeze_bert_layers(self, n_layers: int):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"[NER] 冻结了 SciBERT 的前 {n_layers} 层")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        lstm_output, _ = self.bilstm(sequence_output)
        logits = self.classifier(lstm_output)

        result = {"logits": logits}

        if labels is not None:
            mask = attention_mask.bool()
            loss = -self.crf(logits, labels, mask=mask, reduction="mean")
            result["loss"] = loss

        return result

    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[List[str]]:
        """维特比解码，返回标签序列"""
        self.eval()
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.bilstm(sequence_output)
        logits = self.classifier(lstm_output)
        mask = attention_mask.bool()
        predictions = self.crf.decode(logits, mask=mask)
        return predictions
