"""
论文摘要与审稿生成封装类 (供Agent层直接调用)

提供 generate_summary() 标准接口:
    输入论文文本 + 抽取三元组 → 结构化摘要 + 审稿意见草稿
"""
import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import AutoTokenizer

from .extractive import ExtractiveSummarizer
from .generative import BARTSummarizer
from .review_generator import ReviewGenerator


STRUCTURED_SUMMARY_TEMPLATE = {
    "background": "",
    "contributions": [],
    "methodology": "",
    "experiments": "",
    "results": "",
    "limitations": ""
}


class PaperSummarizer:
    """论文摘要与审稿生成器"""

    def __init__(
        self,
        extractive_model_name: str = "allenai/scibert_scivocab_uncased",
        generative_model_path: Optional[str] = None,
        generative_model_name: str = "facebook/bart-base",
        max_source_length: int = 512,
        max_target_length: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # 抽取式 (TextRank + SciBERT)
        self.extractive = ExtractiveSummarizer(
            model_name=extractive_model_name, device=device
        )

        # 抽象式 (BART, 可选微调模型)
        if generative_model_path:
            self.generative = BARTSummarizer(model_name=generative_model_name)
            ckpt = torch.load(generative_model_path, map_location=device, weights_only=False)
            self.generative.load_state_dict(ckpt["model_state_dict"])
            self.generative.to(device)
            self.generative.eval()
            print(f"[PaperSummarizer] 已加载微调BART: {generative_model_path}")
        else:
            self.generative = BARTSummarizer(model_name=generative_model_name)
            self.generative.to(device)
            self.generative.eval()
            print(f"[PaperSummarizer] 使用预训练BART (未微调)")

        self.gen_tokenizer = AutoTokenizer.from_pretrained(generative_model_name)
        self.review_generator = ReviewGenerator()

    def generate_summary(self, paper_text: str, entities: List[Dict] = None,
                         triples: List[Dict] = None, paper_info: Dict = None) -> Dict:
        """
        Agent调用接口: 生成结构化摘要和审稿意见草稿

        Args:
            paper_text: 论文全文文本
            entities: (可选) 模块二抽取的实体列表
            triples: (可选) 模块二抽取的关系三元组
            paper_info: (可选) 模块一分类结果

        Returns:
            {
                "structured_summary": {...},
                "extractive_skeleton": [...],
                "abstractive_summary": "...",
                "review_draft": {...}
            }
        """
        entities = entities or []
        triples = triples or []
        paper_info = paper_info or {}

        # 1. 抽取式骨架
        key_sentences = self.extractive.summarize(paper_text, top_k=5)

        # 2. 抽象式生成
        abstractive = self._generate_abstractive(paper_text)

        # 3. 结构化摘要
        structured = self._build_structured_summary(paper_text, key_sentences, abstractive)

        # 4. 审稿意见
        review = self.review_generator.generate(entities, triples, paper_info)

        return {
            "structured_summary": structured,
            "extractive_skeleton": key_sentences,
            "abstractive_summary": abstractive,
            "review_draft": review
        }

    @torch.no_grad()
    def _generate_abstractive(self, text: str) -> str:
        """用BART生成抽象式摘要"""
        encoding = self.gen_tokenizer(
            text[:2000],  # 截断以避免超长
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        try:
            outputs = self.generative.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_target_length,
                num_beams=4
            )
            summary = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"[PaperSummarizer] 生成失败: {e}")
            return ""

    def _build_structured_summary(self, text: str, key_sentences: List[str],
                                  abstractive: str) -> Dict:
        """根据抽取骨架和生成摘要构建结构化摘要"""
        structured = dict(STRUCTURED_SUMMARY_TEMPLATE)

        if key_sentences:
            structured["background"] = key_sentences[0] if len(key_sentences) > 0 else ""
            structured["methodology"] = key_sentences[1] if len(key_sentences) > 1 else key_sentences[0]
            structured["results"] = key_sentences[-1] if len(key_sentences) > 1 else ""
            structured["contributions"] = [s[:120] for s in key_sentences[:3]]
            structured["experiments"] = key_sentences[-2] if len(key_sentences) > 2 else ""

        # 用抽象式摘要补充 limitations
        if abstractive:
            structured["limitations"] = abstractive[:200]

        return structured


def generate_summary(paper_text: str, summarizer: "PaperSummarizer",
                     entities: List[Dict] = None, triples: List[Dict] = None,
                     paper_info: Dict = None) -> Dict:
    """Agent调用接口"""
    return summarizer.generate_summary(paper_text, entities, triples, paper_info)
