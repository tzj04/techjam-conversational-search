"""Run the public evaluator against either agent without touching evaluator files.

Usage:
    python3 -m tools.run_eval --agent submission --output results_stage1.json
    python3 -m tools.run_eval --agent starter --output results_starter.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="TechJam local evaluation runner")
    parser.add_argument("--agent", choices=["starter", "submission"], default="submission")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    if args.agent == "starter":
        from starter.agent import Agent
    else:
        from submission.agent import Agent

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    result = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
