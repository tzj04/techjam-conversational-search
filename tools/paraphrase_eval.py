"""Paraphrase stress harness (plan §6 Stage 1.5): the real evaluator's session
loop, except every customer message (opening, replies, override) is passed
through a deterministic paraphraser before the agent sees it. Scoring is
identical to evaluator.local_evaluator — this is our private-set proxy.

Local analysis tool only — never part of the submission bundle.

Usage:
    python3 -m tools.paraphrase_eval --agent starter.agent --level L1 \
        --output results_paraphrase_starter_L1.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import uuid
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)
from tools.paraphraser import Paraphraser


def paraphrase_evaluate(agent, samples, catalog_ids, categories, products, paraphraser) -> dict:
    sessions: list[dict] = []
    for sample in samples:
        session_id = f"para_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
        hit_turn: int | None = None
        best_rank: int | None = None
        for turn in range(1, MAX_TURNS + 1):
            sent_message = paraphraser.paraphrase(user_message, str(sample["sample_id"]), turn)
            try:
                response = agent.respond(session_id, sent_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample, response.get("ask_attribute"), disclosed, boundary_used
                )
        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })

    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paraphrase stress harness (Stage 1.5)")
    parser.add_argument("--agent", default="starter.agent", help="module path exporting Agent")
    parser.add_argument("--level", default="L1", choices=["none", "L1", "L2"])
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results_paraphrase.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    agent_cls = importlib.import_module(args.agent).Agent
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    paraphraser = Paraphraser(level=args.level, seed=args.seed)
    result = paraphrase_evaluate(agent_cls(args.catalog), samples, catalog_ids, categories, products, paraphraser)
    result = {"agent": args.agent, "level": args.level, "seed": args.seed, **result}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
