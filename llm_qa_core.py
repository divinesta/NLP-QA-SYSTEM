"""
Shared utilities for the LLM Question-Answering system.

This module encapsulates:
  * basic question preprocessing (lowercasing, tokenization, punctuation removal)
  * prompt construction
  * interaction with the configured Large Language Model (LLM) API
"""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass
from typing import List, Optional

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - openai is optional at runtime
    OpenAI = None  # type: ignore


@dataclass
class ProcessedQuestion:
    """Container for the artifacts produced by preprocessing."""

    original: str
    lowered: str
    tokens: List[str]
    processed_text: str


class QuestionProcessor:
    """Applies a deterministic, inspectable preprocessing pipeline."""

    token_pattern = re.compile(r"\b\w+\b", re.UNICODE)

    def preprocess(self, question: str) -> ProcessedQuestion:
        cleaned = question.strip()
        lowered = cleaned.lower()
        tokens = self.token_pattern.findall(lowered)
        processed_text = " ".join(tokens)
        return ProcessedQuestion(
            original=cleaned,
            lowered=lowered,
            tokens=tokens,
            processed_text=processed_text,
        )


class LLMClient:
    """Thin wrapper around the underlying LLM provider."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.2,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_BASE_URL")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", temperature))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and OpenAI is not None)

    def generate_response(self, prompt: str) -> str:
        if not self.is_configured:
            return self._offline_response(prompt)

        try:
            # client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            response = client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            text_chunks = [
                block.text.value
                for block in response.output
                if hasattr(block, "text") and getattr(block.text, "value", None)
            ]
            return "\n".join(text_chunks).strip() or self._offline_response(prompt)
        except Exception as exc:  # pragma: no cover - network dependent
            return f"[LLM Error] Unable to fetch response: {exc}"

    @staticmethod
    def _offline_response(prompt: str) -> str:
        preview = prompt.splitlines()[-1]
        return textwrap.dedent(
            f"""
            [Offline Response]
            Unable to reach the configured LLM provider. Please verify that an API key
            is available (set OPENAI_API_KEY or LLM_API_KEY). Last prompt line:
            \"{preview[:120]}...\"
            """
        ).strip()


class LLMQAService:
    """High-level service that couples preprocessing with the LLM client."""

    def __init__(
        self,
        *,
        processor: Optional[QuestionProcessor] = None,
        client: Optional[LLMClient] = None,
    ) -> None:
        self.processor = processor or QuestionProcessor()
        self.client = client or LLMClient()

    def build_prompt(self, processed: ProcessedQuestion) -> str:
        return textwrap.dedent(
            f"""
            You are a helpful university-level teaching assistant. Answer the user's
            question clearly and concisely. Cite key facts when appropriate and
            mention assumptions if the question lacks detail.

            Original question: {processed.original}
            Normalized question: {processed.processed_text}

            Answer:
            """
        ).strip()

    def answer_question(self, question: str) -> dict:
        processed = self.processor.preprocess(question)
        prompt = self.build_prompt(processed)
        answer = self.client.generate_response(prompt)
        return {
            "original_question": processed.original,
            "processed_question": processed.processed_text,
            "tokens": processed.tokens,
            "prompt": prompt,
            "answer": answer,
        }


__all__ = [
    "LLMClient",
    "LLMQAService",
    "ProcessedQuestion",
    "QuestionProcessor",
]

