#!/usr/bin/env python3
"""
Standalone model smoke test using the prompt pattern from models/openai_models.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Ensure CATDA package imports resolve when running from the repo root.
_ROOT = Path(__file__).resolve().parents[1]
_PARENT = _ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from CATDA.models.models import get_model


def main() -> int:
    parser = argparse.ArgumentParser(description="CATDA model smoke test.")
    parser.add_argument(
        "--model",
        default="openai_gpt-5-mini",
        help="Provider_modelname (e.g., openai_gpt-5-mini, openrouter_openai/gpt-4o-mini).",
    )
    parser.add_argument(
        "--country",
        default="France",
        help="Country to ask about (default: France).",
    )
    args = parser.parse_args()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant. Answer the users query accurately."),
            ("human", "What is the capital of {country}?"),
        ]
    )
    chain = prompt | get_model(args.model) | StrOutputParser()
    response = chain.invoke({"country": args.country})
    print(f"Response: {response}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
