"""Tests for LLM module: prompts + client."""
import json
from unittest.mock import MagicMock, patch

import pytest


class TestPrompts:
    """Tests for prompt templates."""

    @pytest.fixture
    def prompts(self):
        from src.llm.prompts import CLASSIFY_PROMPT, EXTRACT_PROMPT, SUMMARIZE_PROMPT, CHECK_MANIFEST_PROMPT
        return {
            "classify": CLASSIFY_PROMPT,
            "extract": EXTRACT_PROMPT,
            "summarize": SUMMARIZE_PROMPT,
            "check_manifest": CHECK_MANIFEST_PROMPT,
        }

    def test_classify_prompt_includes_text(self, prompts):
        result = prompts["classify"].format(paper_text="Test paper text")
        assert "Test paper text" in result
        assert "JSON" in result
        assert "domain" in result.lower() or "领域" in result

    def test_extract_prompt_includes_text(self, prompts):
        result = prompts["extract"].format(paper_text="Test paper text")
        assert "Test paper text" in result
        assert "JSON" in result
        assert "entity" in result.lower() or "实体" in result

    def test_summarize_prompt_includes_text(self, prompts):
        result = prompts["summarize"].format(paper_text="Test paper text")
        assert "Test paper text" in result
        assert "JSON" in result

    def test_check_manifest_prompt_includes_inputs(self, prompts):
        result = prompts["check_manifest"].format(
            paper_text="Test paper",
            entities_json="[]",
            summary_json="{}",
        )
        assert "Test paper" in result
        assert "JSON" in result
        assert "checklist" in result.lower() or "检查" in result


class TestLLMClient:
    """Tests for DeepSeek LLM client with mocked API."""

    @pytest.fixture
    def client(self, mock_openai):
        from src.llm.client import LLMClient
        return LLMClient(api_key="sk-test", base_url="https://api.test.com")

    @pytest.fixture
    def mock_openai(self):
        with patch("src.llm.client.OpenAI") as mock:
            yield mock

    def test_classify_returns_structured_dict(self, client, mock_openai):
        mock_instance = mock_openai.return_value
        mock_instance.chat.completions.create.return_value = _make_response({
            "domains": ["Natural Language Processing", "Machine Learning"],
            "method_type": "Empirical",
        })

        result = client.classify("Bert paper text")
        assert "domains" in result
        assert "method_type" in result
        assert isinstance(result["domains"], list)

    def test_extract_entities_returns_list(self, client, mock_openai):
        mock_instance = mock_openai.return_value
        mock_instance.chat.completions.create.return_value = _make_response({
            "entities": [
                {"text": "BERT", "type": "MODEL"},
                {"text": "SQuAD", "type": "DATASET"},
            ]
        })

        result = client.extract_entities("Bert on SQuAD paper")
        assert "entities" in result
        assert len(result["entities"]) == 2
        assert result["entities"][0]["text"] == "BERT"

    def test_summarize_returns_structured_dict(self, client, mock_openai):
        mock_instance = mock_openai.return_value
        mock_instance.chat.completions.create.return_value = _make_response({
            "background": "Deep learning has...",
            "contributions": ["Contribution 1"],
            "methodology": "We propose...",
            "experiments": "We evaluate on...",
            "results": "Our method achieves...",
            "limitations": "The main limitation is...",
        })

        result = client.summarize("Paper about deep learning")
        assert "background" in result
        assert "contributions" in result
        assert "methodology" in result

    def test_check_manifest_returns_list(self, client, mock_openai):
        mock_instance = mock_openai.return_value
        mock_instance.chat.completions.create.return_value = _make_response([
            {"category": "数据集多样性", "status": "partial", "detail": "使用了2个数据集"},
            {"category": "消融实验", "status": "missing", "detail": "未发现消融实验"},
        ])

        result = client.check_manifest("Paper text", [], {})
        assert isinstance(result, list)
        assert result[0]["category"] == "数据集多样性"

    def test_api_retry_on_failure(self, client, mock_openai):
        mock_instance = mock_openai.return_value
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("API timeout")
            return _make_response({"domains": ["NLP"], "method_type": "Empirical"})

        mock_instance.chat.completions.create.side_effect = side_effect
        result = client.classify("Test")
        assert result["domains"] == ["NLP"]
        assert call_count[0] == 2


def _make_response(data):
    """Build a mock OpenAI chat completion response."""
    content = json.dumps(data, ensure_ascii=False)
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response
