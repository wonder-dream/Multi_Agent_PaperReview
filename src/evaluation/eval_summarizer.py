"""Evaluate summarizer using ROUGE and BERTScore."""
from rouge_score import rouge_scorer


def evaluate_summarizer(summarizer, references: list, texts: list) -> dict:
    """Evaluate extractive summarizer against reference summaries.

    Args:
        summarizer: TextRankSummarizer instance
        references: list of reference summary strings
        texts: list of full paper texts

    Returns:
        dict with ROUGE-1/2/L scores
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for text, ref in zip(texts, references):
        generated = summarizer.summarize(text, num_sentences=3)
        generated_text = " ".join(generated)
        s = scorer.score(ref, generated_text)
        scores["rouge1"].append(s["rouge1"].fmeasure)
        scores["rouge2"].append(s["rouge2"].fmeasure)
        scores["rougeL"].append(s["rougeL"].fmeasure)

    return {
        "rouge1": sum(scores["rouge1"]) / len(scores["rouge1"]) if scores["rouge1"] else 0,
        "rouge2": sum(scores["rouge2"]) / len(scores["rouge2"]) if scores["rouge2"] else 0,
        "rougeL": sum(scores["rougeL"]) / len(scores["rougeL"]) if scores["rougeL"] else 0,
    }
