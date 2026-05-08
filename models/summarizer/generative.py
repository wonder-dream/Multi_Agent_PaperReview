"""
抽象式摘要生成器: BART / T5 Seq2Seq 模型

基于 BART-large 微调科研文本摘要。
支持 HuggingFace BART 和 T5 系列模型。

训练时可使用 SciTLDR 数据微调，推理时通过 generate_summary 接口调用。
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from typing import Dict, Optional


class BARTSummarizer(nn.Module):
    """BART Seq2Seq 摘要生成器"""

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        max_target_length: int = 128,
        freeze_encoder_layers: int = 0,
        freeze_decoder_layers: int = 0
    ):
        super().__init__()
        self.model_name = model_name
        self.max_target_length = max_target_length
        self.config = AutoConfig.from_pretrained(model_name)
        self.bart = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if freeze_encoder_layers > 0:
            self._freeze_layers(self.bart.model.encoder, freeze_encoder_layers)
        if freeze_decoder_layers > 0:
            self._freeze_layers(self.bart.model.decoder, freeze_decoder_layers)

        print(f"[BARTSummarizer] 加载 {model_name}, "
              f"参数量: {sum(p.numel() for p in self.bart.parameters()) / 1e6:.2f}M")

    def _freeze_layers(self, module, n_layers):
        for param in module.embed_positions.parameters():
            param.requires_grad = False
        for i, layer in enumerate(module.layers):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"[BARTSummarizer] 冻结了前 {n_layers} 层")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        result = {"loss": outputs.loss, "logits": outputs.logits}
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        early_stopping: bool = True
    ) -> torch.Tensor:
        """生成摘要"""
        self.eval()
        outputs = self.bart.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        return outputs
