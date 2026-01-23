#!/usr/bin/env python3
"""
Smoke test for core CATDA flows using a live LLM.

Default: model-only invoke with usage capture (mimics extract_main usage tracking).
Optional: full agent run (requires Neo4j + vector deps + running DB).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

# Ensure CATDA package imports resolve when running from the repo root.
_ROOT = Path(__file__).resolve().parents[1]
_PARENT = _ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from langchain_core.callbacks.usage import get_usage_metadata_callback

from CATDA.models.models import get_model
from CATDA.service.agent_setup import setup_agent
from CATDA.service.agent_wrapper import run_agent


def _provider_from_model(model_name: str) -> str:
    return (model_name.split("_", 1)[0] if "_" in model_name else model_name).lower()


def _has_provider_key(provider: str) -> Tuple[bool, str]:
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY")), "OPENAI_API_KEY"
    if provider == "google":
        return bool(os.getenv("GOOGLE_API_KEY")), "GOOGLE_API_KEY"
    if provider == "deepseek":
        return bool(os.getenv("DEEPSEEK_API_KEY")), "DEEPSEEK_API_KEY"
    return True, ""


def run_model_smoke(model_name: str, prompt: str) -> int:
    provider = _provider_from_model(model_name)
    ok, key = _has_provider_key(provider)
    if not ok:
        print(f"SKIP: {key} not set for provider '{provider}'.")
        return 0

    model = get_model(model_name)
    with get_usage_metadata_callback() as usage_cb:
        response = model.invoke(prompt)

    content = getattr(response, "content", response)
    print("LLM response:")
    print(content)
    print("\nUsage metadata:")
    print(usage_cb.usage_metadata)
    return 0


def run_agent_smoke(model_name: str) -> int:
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not neo4j_password:
        print("SKIP: NEO4J_PASSWORD not set for agent smoke test.")
        return 0

    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")

    agent = setup_agent(
        model_name=model_name,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        verbose=False,
    )

    result = run_agent(agent, "Return only the number 1.", [])
    print("Agent response:")
    print(result.get("answer"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CATDA smoke test (LLM + optional agent).")
    parser.add_argument(
        "--model",
        default=os.getenv("CATDA_TEST_MODEL", "openai_gpt-5-mini"),
        help="Provider_modelname (default: openai_gpt-5-mini).",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with a short JSON object: {\"ok\": true}.",
        help="Prompt for the model-only smoke test.",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Also run the full agent smoke test (requires Neo4j and vector deps).",
    )
    args = parser.parse_args()

    run_model_smoke(args.model, args.prompt)
    if args.agent:
        run_agent_smoke(args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
