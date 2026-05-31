"""DeepSeek API client for structured paper analysis."""
import json
import time
from typing import Dict, List

from openai import OpenAI

from .prompts import CLASSIFY_PROMPT, EXTRACT_PROMPT, SUMMARIZE_PROMPT, CHECK_MANIFEST_PROMPT


class LLMClient:
    """Calls DeepSeek API (OpenAI-compatible) for paper analysis tasks."""

    def __init__(
        self,
        api_key: str = "sk-placeholder",
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        max_retries: int = 2,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries

    def classify(self, paper_text: str) -> Dict:
        prompt = CLASSIFY_PROMPT.format(paper_text=paper_text)
        return self._call(prompt)

    def extract_entities(self, paper_text: str) -> Dict:
        prompt = EXTRACT_PROMPT.format(paper_text=paper_text)
        return self._call(prompt)

    def summarize(self, paper_text: str) -> Dict:
        prompt = SUMMARIZE_PROMPT.format(paper_text=paper_text)
        return self._call(prompt)

    def check_manifest(
        self, paper_text: str, entities: List | Dict, summary: Dict
    ) -> List:
        prompt = CHECK_MANIFEST_PROMPT.format(
            paper_text=paper_text,
            entities_json=json.dumps(entities, ensure_ascii=False, indent=2),
            summary_json=json.dumps(summary, ensure_ascii=False, indent=2),
        )
        return self._call(prompt)

    def _call(self, prompt: str) -> Dict | List:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise scientific document analyzer. Always output valid JSON exactly as requested."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )
                content = response.choices[0].message.content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
                return json.loads(content)
            except (json.JSONDecodeError, Exception):
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return {} if "classify" in prompt or "extract" in prompt or "summarize" in prompt else []
