"""Tests for classifier module: model + dataset."""
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestMultiTaskClassifier:
    """Tests for SciBERTMultiTaskClassifier."""

    @pytest.fixture
    def model(self):
        from src.classifier.model import SciBERTMultiTaskClassifier
        return SciBERTMultiTaskClassifier(
            num_domains=4,
            num_methods=4,
            pretrained=False,
            hidden_size=128,
        )

    def test_model_creates_successfully(self, model):
        assert model is not None
        assert model.num_domains == 4
        assert model.num_methods == 4

    def test_forward_pass_shapes(self, model):
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, 30000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        output = model(input_ids, attention_mask)
        assert "domain_logits" in output
        assert "method_logits" in output
        assert output["domain_logits"].shape == (batch_size, 4)
        assert output["method_logits"].shape == (batch_size, 4)

    def test_predict_returns_labels(self, model):
        input_ids = torch.randint(0, 30000, (1, 64))
        attention_mask = torch.ones(1, 64, dtype=torch.long)

        result = model.predict(input_ids, attention_mask)
        assert "domains" in result
        assert "method_type" in result
        assert isinstance(result["domains"], list)
        assert isinstance(result["method_type"], str)

    def test_predict_text_interface(self, model):
        """predict_text handles string input."""
        tokens = {"input_ids": torch.randint(0, 30000, (1, 64)),
                  "attention_mask": torch.ones(1, 64, dtype=torch.long)}
        result = model.predict(tokens["input_ids"], tokens["attention_mask"])
        assert "domains" in result
        assert "method_type" in result


class TestPeerReadDataset:
    """Tests for PeerRead dataset using a mock tokenizer."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Return a mock that produces valid tensor outputs."""
        mock = MagicMock()
        mock.return_value = {
            "input_ids": torch.randint(0, 30000, (1, 512)),
            "attention_mask": torch.ones(1, 512, dtype=torch.long),
        }
        return mock

    def test_dataset_len(self, mock_tokenizer):
        from src.classifier.dataset import PeerReadDataset

        samples = [
            {"text": "Paper A", "domains": ["NLP"], "method_type": "Empirical"},
            {"text": "Paper B", "domains": ["CV"], "method_type": "Theoretical"},
        ]

        with patch("src.classifier.dataset.AutoTokenizer") as mock_at:
            mock_at.from_pretrained.return_value = mock_tokenizer
            dataset = PeerReadDataset(samples)
            assert len(dataset) == 2

    def test_domain_label_mapping(self):
        from src.classifier.dataset import DOMAIN_LABELS

        assert "NLP" in DOMAIN_LABELS
        assert len(DOMAIN_LABELS) == 4

    def test_method_label_mapping(self):
        from src.classifier.dataset import METHOD_LABELS

        assert "Empirical" in METHOD_LABELS
        assert len(METHOD_LABELS) == 4

    def test_item_shape(self, mock_tokenizer):
        from src.classifier.dataset import PeerReadDataset

        samples = [{"text": "Paper A", "domains": ["NLP", "ML"], "method_type": "Empirical"}]

        with patch("src.classifier.dataset.AutoTokenizer") as mock_at:
            mock_at.from_pretrained.return_value = mock_tokenizer
            dataset = PeerReadDataset(samples)
            item = dataset[0]
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "domain_labels" in item
            assert "method_label" in item
            assert item["domain_labels"].shape[0] == 4
