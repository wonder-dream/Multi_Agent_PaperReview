"""Tests for summarizer module: TextRank + checklist engine."""
import pytest


class TestTextRank:
    """Tests for TextRank extractive summarizer."""

    @pytest.fixture
    def summarizer(self):
        from src.summarizer.textrank import TextRankSummarizer
        return TextRankSummarizer()

    def test_summarize_returns_nonempty(self, summarizer):
        text = (
            "Deep learning has transformed natural language processing. "
            "BERT introduced bidirectional pre-training. "
            "RoBERTa improved upon BERT with more data. "
            "Experiments show RoBERTa outperforms BERT on GLUE. "
            "Ablation studies confirm the importance of dynamic masking. "
            "Future work will explore larger models."
        )
        result = summarizer.summarize(text, num_sentences=3)
        assert len(result) == 3
        assert all(isinstance(s, str) for s in result)
        assert all(len(s) > 0 for s in result)

    def test_summarize_returns_fewer_when_short_text(self, summarizer):
        text = "Short text. Only two sentences."
        result = summarizer.summarize(text, num_sentences=5)
        assert len(result) <= 2

    def test_empty_text_returns_empty(self, summarizer):
        assert summarizer.summarize("") == []
        assert summarizer.summarize("   ") == []

    def test_structured_summary(self, summarizer):
        text = (
            "Background: NLP is important. "
            "We propose a new BERT-based method. "
            "We use SQuAD and GLUE datasets. "
            "Our method achieves F1 of 92.5 on SQuAD. "
            "One limitation is the model size."
        )
        result = summarizer.structured_summary(text)
        assert "background" in result
        assert "contributions" in result
        assert "methodology" in result
        assert "experiments" in result
        assert "results" in result
        assert "limitations" in result


class TestChecklistEngine:
    """Tests for completeness checklist generation."""

    @pytest.fixture
    def engine(self):
        from src.summarizer.checklist import ChecklistEngine
        return ChecklistEngine()

    def test_generates_items_from_entities(self, engine):
        entities = [
            {"text": "BERT", "type": "MODEL"},
            {"text": "SQuAD", "type": "DATASET"},
            {"text": "F1", "type": "METRIC"},
        ]
        items = engine.generate(entities, paper_text="We use BERT on SQuAD.")
        assert isinstance(items, list)
        assert len(items) > 0
        for item in items:
            assert "category" in item
            assert "status" in item
            assert "detail" in item

    def test_detects_missing_ablation(self, engine):
        entities = [{"text": "BERT", "type": "MODEL"}]
        text = "We train BERT on our data."
        items = engine.generate(entities, paper_text=text)
        ablation_items = [i for i in items if "消融" in i["category"]]
        assert len(ablation_items) >= 1

    def test_detects_missing_code(self, engine):
        entities = [{"text": "BERT", "type": "MODEL"}]
        text = "We use BERT."
        items = engine.generate(entities, paper_text=text)
        code_items = [i for i in items if "代码" in i["category"]]
        assert len(code_items) >= 1

    def test_ethics_check(self, engine):
        text_with_ethics = "We discuss ethical implications of our work."
        text_without = "We train a model on SQuAD."
        items_with = engine.generate([], paper_text=text_with_ethics)
        items_without = engine.generate([], paper_text=text_without)

        ethics_with = [i["status"] for i in items_with if "伦理" in i["category"]]
        ethics_without = [i["status"] for i in items_without if "伦理" in i["category"]]
        if ethics_with and ethics_without:
            assert ethics_with[0] != ethics_without[0]
