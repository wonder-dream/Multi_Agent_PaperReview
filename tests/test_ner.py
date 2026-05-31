"""Tests for NER module: BiLSTM-CRF model + SciERC dataset."""
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestBiLSTMCRF:
    """Tests for SciBERT + BiLSTM + CRF NER model."""

    @pytest.fixture
    def model(self):
        from src.ner.model import BiLSTMCRFNER
        return BiLSTMCRFNER(
            num_labels=11,
            pretrained=False,
            hidden_size=128,
            lstm_hidden=128,
        )

    def test_model_creates(self, model):
        assert model.num_labels == 11
        assert model.labels[0] == "O"

    def test_forward_pass_shapes(self, model):
        batch, seq = 2, 32
        input_ids = torch.randint(0, 30000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)

        emissions, mask = model.forward(input_ids, attention_mask)
        assert emissions.shape == (batch, seq, 11)
        assert mask.shape == (batch, seq)

    def test_decode_returns_label_sequences(self, model):
        batch, seq = 2, 32
        input_ids = torch.randint(0, 30000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)

        tags = model.decode(input_ids, attention_mask)
        assert isinstance(tags, list)
        assert len(tags) == batch
        assert all(isinstance(t, list) for t in tags)
        assert all(len(t) == seq for t in tags)
        assert all(isinstance(l, str) for t in tags for l in t)

    def test_predict_extracts_entities(self, model):
        """predict() converts tag sequences to entity dicts."""
        input_ids = torch.randint(0, 30000, (1, 32))
        attention_mask = torch.ones(1, 32, dtype=torch.long)

        result = model.predict(input_ids, attention_mask)
        assert "entities" in result
        assert isinstance(result["entities"], list)


class TestSciERCDataset:
    """Tests for SciERC dataset loader."""

    def test_label_mappings(self):
        from src.ner.dataset import ENTITY_TYPES, LABEL2ID, ID2LABEL

        assert len(ENTITY_TYPES) == 5
        assert "MODEL" in ENTITY_TYPES
        assert "DATASET" in ENTITY_TYPES
        assert "METRIC" in ENTITY_TYPES
        assert LABEL2ID["O"] == 0

    def test_dataset_loads_scierc_format(self):
        from src.ner.dataset import SciERCDataset

        samples = [
            {
                "tokens": ["We", "use", "BERT", "on", "SQuAD", "."],
                "entities": [
                    {"text": "BERT", "type": "MODEL", "start": 2, "end": 3},
                    {"text": "SQuAD", "type": "DATASET", "start": 4, "end": 5},
                ],
            },
        ]

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.__getitem__ = lambda self, k: torch.ones(1).long()

        with patch("src.ner.dataset.AutoTokenizer") as mock_at:
            mock_at.from_pretrained.return_value = mock_tokenizer
            dataset = SciERCDataset(samples)
            assert len(dataset) == 1
