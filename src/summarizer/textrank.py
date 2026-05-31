"""TextRank extractive summarization with SciBERT sentence embeddings."""
import re
from typing import List

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


class TextRankSummarizer:
    """Extractive summarizer using TextRank + SciBERT sentence embeddings."""

    def __init__(self, embedding_model: str = None):
        self.embedding_model_name = embedding_model
        self._model = None

    def _get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get sentence embeddings. Uses TF-IDF as default, SciBERT if model is configured."""
        if self.embedding_model_name is not None:
            try:
                from sentence_transformers import SentenceTransformer
                if self._model is None:
                    self._model = SentenceTransformer(self.embedding_model_name)
                return self._model.encode(sentences)
            except Exception:
                pass
        return self._tfidf_embeddings(sentences)

    def _tfidf_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Fallback: simple TF-IDF sentence vectors."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english")
        return vec.fit_transform(sentences).toarray()

    def summarize(self, text: str, num_sentences: int = 5, diversity: float = 0.3) -> List[str]:
        """Extract top-k sentences using TextRank + MMR."""
        sentences = _split_sentences(text)
        if not sentences:
            return []
        if len(sentences) <= num_sentences:
            return sentences

        embeddings = self._get_embeddings(sentences)
        sim_matrix = cosine_similarity(embeddings)

        # PageRank
        G = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(G, max_iter=100)

        # MMR selection
        selected = []
        candidates = list(range(len(sentences)))
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)

        for _ in range(num_sentences):
            if not ranked:
                break
            best = ranked[0]
            selected.append(best)
            ranked.remove(best)

            # MMR re-rank remaining
            if ranked:
                mmr_scores = []
                for idx in ranked:
                    relevance = scores[idx]
                    redundancy = max(sim_matrix[idx][s] for s in selected) if selected else 0
                    mmr_scores.append(relevance - diversity * redundancy)
                ranked = [c for _, c in sorted(zip(mmr_scores, ranked), reverse=True)]

        return [sentences[i] for i in sorted(selected)]

    def structured_summary(self, text: str) -> dict:
        """Generate a structured summary with 6 standard sections."""
        sentences = self.summarize(text, num_sentences=8)
        return {
            "background": sentences[0] if len(sentences) > 0 else "",
            "contributions": [s for s in sentences if any(
                kw in s.lower() for kw in ["propose", "introduce", "present", "novel", "contribute"])],
            "methodology": sentences[1] if len(sentences) > 1 else "",
            "experiments": sentences[2] if len(sentences) > 2 else "",
            "results": sentences[3] if len(sentences) > 3 else "",
            "limitations": sentences[-1] if len(sentences) > 1 else "",
        }


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    text = re.sub(r"\n+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]
