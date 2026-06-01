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
            pretrained=False,
            hidden_size=128,
            lstm_hidden=128,
        )

    def test_freeze_layers_freezes_bottom_bert_layers(self):
        """freeze_layers=N freezes embeddings and bottom N encoder layers."""
        from src.ner.model import BiLSTMCRFNER

        m = BiLSTMCRFNER(pretrained=False, hidden_size=128, lstm_hidden=128, freeze_layers=1)
        # 2-layer tiny bert; freeze_layers=1 freezes embeddings + layer 0
        for name, p in m.named_parameters():
            if "embeddings" in name or "encoder.layer.0" in name:
                assert not p.requires_grad, f"{name} should be frozen"
            elif "encoder.layer.1" in name:
                assert p.requires_grad, f"{name} should be trainable"
        # BiLSTM and classifier should always be trainable
        bilstm_trainable = any(p.requires_grad for _, p in m.bilstm.named_parameters())
        assert bilstm_trainable, "BiLSTM should be trainable"

    def test_default_params_reflect_reduced_capacity(self):
        """Default dropout=0.4, lstm_hidden=256 for regularized config."""
        from src.ner.model import BiLSTMCRFNER

        m = BiLSTMCRFNER(pretrained=False, hidden_size=128)
        assert m.dropout.p == 0.4
        assert m.bilstm.hidden_size == 256

    def test_model_creates(self, model):
        assert model.num_labels == 9
        assert model.labels[0] == "O"

    def test_forward_pass_shapes(self, model):
        batch, seq = 2, 32
        input_ids = torch.randint(0, 30000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)

        emissions, mask = model.forward(input_ids, attention_mask)
        assert emissions.shape[0] == batch
        assert emissions.shape[2] == 9
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

    def test_tags_to_entities_with_tokenizer(self):
        """_tags_to_entities decodes entity text when tokenizer is given."""
        from src.ner.model import _tags_to_entities
        from transformers import AutoTokenizer

        tags = ["O", "B-METHOD", "I-METHOD", "O", "B-TASK", "O"]
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        input_ids = tokenizer("We use deep learning for classification", add_special_tokens=False)["input_ids"]

        entities = _tags_to_entities(tags, input_ids, tokenizer)
        assert len(entities) == 2
        assert entities[0]["type"] == "METHOD"
        assert entities[0]["text"] != ""
        assert entities[1]["type"] == "TASK"
        assert entities[1]["text"] != ""


class TestClassWeighting:
    """Tests for compute_sample_weights — handles METRIC tail-class weighting."""

    def test_metric_samples_get_higher_weight(self):
        from src.ner.dataset import compute_sample_weights

        samples = [
            {"tokens": ["a", "b"], "entities": [{"type": "METHOD", "start": 0, "end": 1}]},
            {"tokens": ["c", "d"], "entities": [{"type": "METRIC", "start": 0, "end": 1}]},
            {"tokens": ["e", "f"], "entities": []},
        ]
        weights = compute_sample_weights(samples, class_weight={"METRIC": 3.0})
        assert len(weights) == 3
        # METRIC sample (index 1) should have highest weight
        assert weights[1] > weights[0]
        assert weights[1] > weights[2]

    def test_no_class_weight_returns_unity(self):
        from src.ner.dataset import compute_sample_weights

        samples = [
            {"tokens": ["a"], "entities": [{"type": "METHOD", "start": 0, "end": 1}]},
            {"tokens": ["b"], "entities": []},
        ]
        weights = compute_sample_weights(samples, class_weight={})
        assert weights == [1.0, 1.0]


class TestSciERCDataset:
    """Tests for SciERC dataset loader."""

    def test_label_mappings(self):
        from src.ner.dataset import ENTITY_TYPES, LABEL2ID, ID2LABEL

        assert len(ENTITY_TYPES) == 4
        assert "TASK" in ENTITY_TYPES
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
