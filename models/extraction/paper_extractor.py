"""
信息抽取封装类 (供Agent层直接调用)

提供 extract_information() 标准接口:
    输入论文文本 → NER识别实体 → 枚举实体对 → RE判断关系 → 输出结构化三元组
"""
import re
import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import AutoTokenizer

from .ner_model import SciBERTNERModel
from .re_model import SciBERTRelationClassifier
from .dataset import NERDataset, REDataset, SPECIAL_TOKENS


class PaperExtractor:
    """论文信息抽取器，封装NER+RE完整pipeline"""

    ENTITY_TYPE_NAMES = {
        "TASK": "TASK", "METHOD": "METHOD", "METRIC": "METRIC",
        "DATASET": "DATASET", "TERM": "TERM", "GENERIC": "GENERIC"
    }

    def __init__(
        self,
        ner_model_path: str,
        re_model_path: str,
        model_name: str = "allenai/scibert_scivocab_uncased",
        max_length: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.max_length = max_length
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.re_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.re_tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

        self.ner_model = SciBERTNERModel(model_name=model_name)
        ner_ckpt = torch.load(ner_model_path, map_location=device, weights_only=False)
        self.ner_model.load_state_dict(ner_ckpt["model_state_dict"])
        self.ner_model.to(device)
        self.ner_model.eval()

        self.re_model = SciBERTRelationClassifier(model_name=model_name)
        re_ckpt = torch.load(re_model_path, map_location=device, weights_only=False)
        self.re_model.load_state_dict(re_ckpt["model_state_dict"])
        self.re_model.to(device)
        self.re_model.eval()

        print(f"[PaperExtractor] NER模型: {ner_model_path}")
        print(f"[PaperExtractor] RE模型: {re_model_path}")

    @torch.no_grad()
    def extract_information(self, paper_text: str) -> Dict:
        """Agent调用接口: 从论文文本抽取实体和关系三元组"""
        sentences = self._split_sentences(paper_text)
        all_entities = []
        all_triples = []

        for sent_id, sent in enumerate(sentences):
            if not sent.strip():
                continue
            entities = self._extract_entities_from_sent(sent)
            offset = sum(len(s.split()) for s in sentences[:sent_id])
            for e in entities:
                e["sent_id"] = sent_id
                e["char_start"] = paper_text.find(e["text"]) if e["text"] in paper_text else -1
                e["char_end"] = e["char_start"] + len(e["text"]) if e["char_start"] >= 0 else -1
            all_entities.extend(entities)

            if len(entities) >= 2:
                triples = self._extract_relations_from_sent(sent, entities)
                for t in triples:
                    t["sent_id"] = sent_id
                all_triples.extend(triples)

        return {
            "entities": [{"text": e["text"], "type": e["type"], "sent_id": e["sent_id"],
                          "char_start": e["char_start"], "char_end": e["char_end"]}
                         for e in all_entities],
            "triples": [{"head": t["head"], "relation": t["relation"], "tail": t["tail"],
                         "sent_id": t["sent_id"]} for t in all_triples]
        }

    @torch.no_grad()
    def _extract_entities_from_sent(self, sentence: str) -> List[Dict]:
        """对单个句子做NER"""
        words = sentence.split()
        if len(words) < 2:
            return []

        encoding = self.tokenizer(
            words, is_split_into_words=True,
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        word_ids = encoding.word_ids()

        predictions = self.ner_model.decode(input_ids, attention_mask)
        tags = [NERDataset.ID2LABEL.get(p, "O") for p in predictions[0]]

        entities = []
        current_entity = None
        for i, (tag, word_id) in enumerate(zip(tags, word_ids)):
            if word_id is None:
                continue
            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"text": words[word_id], "type": tag[2:],
                                  "word_start": word_id, "word_end": word_id}
            elif tag.startswith("I-") and current_entity and current_entity["type"] == tag[2:]:
                current_entity["text"] += " " + words[word_id]
                current_entity["word_end"] = word_id
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        if current_entity:
            entities.append(current_entity)
        return entities

    @torch.no_grad()
    def _extract_relations_from_sent(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        """对句子内实体对做RE"""
        triples = []
        words = sentence.split()
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i == j:
                    continue
                e1, e2 = entities[i], entities[j]
                marked = self._make_marked_sentence(
                    words, e1["word_start"], e1["word_end"],
                    e2["word_start"], e2["word_end"]
                )
                encoding = self.re_tokenizer(
                    marked, max_length=self.max_length, padding="max_length",
                    truncation=True, return_tensors="pt"
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                outputs = self.re_model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(outputs["logits"], dim=-1).item()
                rel_name = REDataset.ID2RELATION.get(pred, "NO-RELATION")
                if rel_name != "NO-RELATION":
                    triples.append({"head": e1["text"], "relation": rel_name, "tail": e2["text"]})
        return triples

    def _make_marked_sentence(self, words, s1, e1, s2, e2):
        e1_first = s1 <= s2
        if e1_first:
            marked = (words[:s1] + ["[E1]"] + words[s1:e1 + 1] + ["[/E1]"]
                      + words[e1 + 1:s2] + ["[E2]"] + words[s2:e2 + 1] + ["[/E2]"]
                      + words[e2 + 1:])
        else:
            marked = (words[:s2] + ["[E2]"] + words[s2:e2 + 1] + ["[/E2]"]
                      + words[e2 + 1:s1] + ["[E1]"] + words[s1:e1 + 1] + ["[/E1]"]
                      + words[e1 + 1:])
        return " ".join(marked)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """简单分句"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]


def extract_information(paper_text: str, extractor: "PaperExtractor") -> Dict:
    """Agent调用接口"""
    return extractor.extract_information(paper_text)
