"""Prompt templates for the 4-stage LLM analysis pipeline."""

CLASSIFY_PROMPT = """\
You are a scientific paper classifier. Read the following paper and output a JSON object with two fields:

1. "domains": a list of research domains (e.g. ["Natural Language Processing", "Computer Vision", "Machine Learning", "Artificial Intelligence"])
2. "method_type": one of "Empirical", "Theoretical", "Survey", or "Benchmark"

Return ONLY the JSON object, no other text.

Paper:
{paper_text}
"""

EXTRACT_PROMPT = """\
You are a scientific entity extractor. Read the following paper and extract all named entities.

Output a JSON object with an "entities" list. Each entity must have:
- "text": the exact entity name as it appears in the paper
- "type": one of MODEL, DATASET, METRIC, METHOD, TASK

Also extract relations between entities where possible. Add a "relations" list with objects:
- "head": entity text
- "relation": one of EVALUATED_ON, ACHIEVES, USES, SOLVES
- "tail": entity text

Return ONLY the JSON object, no other text.

Paper:
{paper_text}
"""

SUMMARIZE_PROMPT = """\
You are a scientific paper summarizer. Read the following paper and produce a structured summary in JSON format with these fields:

- "background": research background and motivation (1-2 sentences)
- "contributions": list of main contributions (bullet points as strings)
- "methodology": core method overview (2-3 sentences)
- "experiments": experimental setup, datasets, and baselines (2-3 sentences)
- "results": main experimental findings (2-3 sentences)
- "limitations": author-stated limitations or weaknesses you can identify

Return ONLY the JSON object, no other text.

Paper:
{paper_text}
"""

CHECK_MANIFEST_PROMPT = """\
You are a scientific paper completeness checker. Based on the paper text, extracted entities, and summary, generate a structured completeness checklist.

Output a JSON list of checklist items. Each item must have:
- "category": one of 数据集多样性, Baseline对比, 消融实验, 代码开源, 伦理声明, 指标完整性
- "status": one of ok, partial, missing, unchecked
- "detail": specific observation or recommendation (1 sentence)

Guidelines:
- Check how many datasets are used (≥3 is good for most domains)
- Check if ablation studies are mentioned
- Check if code or data links are provided
- Check if ethics review or limitations are discussed
- Check if evaluation metrics are comprehensive

Paper text:
{paper_text}

Extracted entities:
{entities_json}

Summary:
{summary_json}

Return ONLY the JSON list, no other text.
"""
