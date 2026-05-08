"""
RE模型: SciBERT + 实体标记 + 关系分类
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Dict, Optional


SPECIAL_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]


class SciBERTRelationClassifier(nn.Module):
    """
    关系分类模型

    输入构造: [CLS] [E1] ent1 [/E1] ... [E2] ent2 [/E2] [SEP]
    通过实体marker平均池化获取实体表示
    拼接 [CLS] + e1 + e2 → Linear → 关系分类
    """

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_relations: int = 8,
        dropout_rate: float = 0.1,
        freeze_bert_layers: int = 0
    ):
        super().__init__()
        self.num_relations = num_relations
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size

        # 添加特殊token用于实体标记
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")

        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 3, num_relations)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _freeze_bert_layers(self, n_layers: int):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"[RE] 冻结了 SciBERT 的前 {n_layers} 层")

    def _pool_at_marker(self, hidden: torch.Tensor, input_ids: torch.Tensor,
                        marker_id: int) -> torch.Tensor:
        """在marker token位置做平均池化"""
        mask = (input_ids == marker_id).float()
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=-1, keepdim=True) + 1e-9)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        cls_vec = hidden[:, 0, :]
        e1_vec = self._pool_at_marker(hidden, input_ids, self.e1_id)
        e2_vec = self._pool_at_marker(hidden, input_ids, self.e2_id)

        combined = torch.cat([cls_vec, e1_vec, e2_vec], dim=-1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        probs = torch.softmax(logits, dim=-1)

        result = {"logits": logits, "probs": probs}
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            result["loss"] = loss

        return result
