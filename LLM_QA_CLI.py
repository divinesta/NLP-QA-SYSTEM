"""
Command-line interface for the LLM Q&A system.

Usage examples:
    python LLM_QA_CLI.py --question "What is NLP?"
    python LLM_QA_CLI.py
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from llm_qa_core import LLMClient, LLMQAService


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask natural-language questions and receive LLM-generated answers.",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Submit a single question (otherwise an interactive session starts).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the default LLM model (falls back to env LLM_MODEL).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override generation temperature (falls back to env LLM_TEMPERATURE).",
    )
    return parser.parse_args(argv)


def render_result(result: dict) -> str:
    tokens_preview = ", ".join(result["tokens"][:10])
    if len(result["tokens"]) > 10:
        tokens_preview += ", ..."
    return (
        f"\nProcessed question: {result['processed_question']}\n"
        f"Tokens ({len(result['tokens'])}): {tokens_preview}\n"
        f"Answer:\n{result['answer']}\n"
    )


def interactive_loop(service: LLMQAService) -> None:
    print("LLM Q&A CLI — type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = service.answer_question(question)
        print(render_result(result))


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    client = LLMClient(model=args.model, temperature=args.temperature or 0.2)
    service = LLMQAService(client=client)

    if args.question:
        result = service.answer_question(args.question)
        print(render_result(result))
        return 0

    interactive_loop(service)
    return 0


if __name__ == "__main__":
    sys.exit(main())

