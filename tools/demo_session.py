"""Replay one public session through the real evaluator loop with the
submission agent, printing the full turn-by-turn exchange (plan §6 Stage 7).

Local demo tool only — never part of the submission bundle.

Usage:
    python3 -m tools.demo_session --sample-id public_0003
"""
from __future__ import annotations

import argparse
import importlib

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)


def describe(product: dict, limit: int = 72) -> str:
    title = str(product.get("title") or "?")
    if len(title) > limit:
        title = title[: limit - 3] + "..."
    price = product.get("price")
    price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "n/a"
    return f"{title}  [{price_str}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one session turn by turn")
    parser.add_argument("--sample-id", default="public_0003")
    parser.add_argument("--agent", default="submission.agent", help="module path exporting Agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    sample = next(s for s in load_jsonl(args.dataset) if s["sample_id"] == args.sample_id)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = importlib.import_module(args.agent).Agent(args.catalog)

    session_id = f"demo_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)

    print(f"session {sample['sample_id']}  scenario={sample['scenario_type']}")
    print(f"target  {target}  {describe(products[target])}")
    print(f"card    hard={card['hard_constraints']}  soft={card['soft_preferences']}")
    print("=" * 78)
    for turn in range(1, MAX_TURNS + 1):
        print(f"\nTurn {turn}")
        print(f"  customer > {user_message}")
        response = agent.respond(session_id, user_message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        print(f"  agent    > {response['message']}  (ask_attribute={response['ask_attribute']!r})")
        if ranked:
            for index, parent_asin in enumerate(ranked, 1):
                marker = "  <-- TARGET" if parent_asin == target else ""
                print(f"      {index:2}. {describe(products.get(parent_asin, {}))}{marker}")
        else:
            print("      (no recommendations returned — gated)")
        if override_applied and target in ranked:
            print(f"\nHIT at turn {turn}, rank {ranked.index(target) + 1}")
            break
        if turn == MAX_TURNS:
            print("\nNO HIT within 10 turns")
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


if __name__ == "__main__":
    main()
