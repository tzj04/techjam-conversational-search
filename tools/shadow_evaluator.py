"""Shadow evaluator: identical session flow to evaluator.local_evaluator, but the
session never stops at the first hit. Records the target's rank in the returned
recommendations at every turn, giving the true rank-vs-turn counterfactual that
the early-turn recommendation-gating idea (plan §5.4) depends on.

Local analysis tool only — never part of the submission bundle.

Usage:
    python3 -m tools.shadow_evaluator --catalog data/catalog.jsonl \
        --dataset data/public_set.jsonl --output results_shadow.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
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
    normalize_recommendations,
)


def shadow_evaluate(agent, samples, catalog_ids, categories, products) -> list[dict]:
    sessions: list[dict] = []
    for sample in samples:
        session_id = f"shadow_{sample['sample_id']}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
        turn_ranks: list[int | None] = []       # rank of target in returned recs, per turn (1-based)
        eligible: list[bool] = []               # override_applied at that turn (hit-eligible)
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            turn_ranks.append(ranked.index(target) + 1 if target in ranked else None)
            eligible.append(override_applied)
            # NOTE: no break on hit — the whole point of the shadow run.
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
        first_hit = next(
            (i + 1 for i, (rank, ok) in enumerate(zip(turn_ranks, eligible)) if ok and rank is not None),
            None,
        )
        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "turn_ranks": turn_ranks,
            "eligible": eligible,
            "baseline_first_hit_turn": first_hit,
            "baseline_rank": turn_ranks[first_hit - 1] if first_hit else None,
        })
    return sessions


def analyze(sessions: list[dict]) -> dict:
    """Answer the gating question: for sessions the baseline hits on turn 1,
    does the target's rank actually improve if the session continues?"""
    turn1 = [s for s in sessions if s["baseline_first_hit_turn"] == 1]
    out: dict = {"turn1_hit_sessions": len(turn1)}

    per_turn_ranks: dict[int, list[int | None]] = {t: [] for t in range(1, 6)}
    for s in turn1:
        for t in range(1, 6):
            per_turn_ranks[t].append(s["turn_ranks"][t - 1] if len(s["turn_ranks"]) >= t else None)
    out["turn1_cohort_mean_rank_by_turn"] = {
        t: round(statistics.fmean(r for r in ranks if r is not None), 3) if any(r is not None for r in ranks) else None
        for t, ranks in per_turn_ranks.items()
    }
    out["turn1_cohort_in_top10_by_turn"] = {
        t: sum(r is not None for r in ranks) for t, ranks in per_turn_ranks.items()
    }

    # Exact score deltas for "withhold recommendations until turn T" applied to the
    # turn-1-hit cohort only, using the real algebra: per-session contribution to
    # TechnicalScore is 0.5*hit + 0.3*RR + 0.02*(11 - hit_turn), averaged over N.
    n_total = len(sessions)
    out["ev_delta_if_defer_turn1_cohort"] = {}
    for defer_to in (2, 3, 4):
        delta = 0.0
        dropped = 0
        for s in turn1:
            r1 = s["turn_ranks"][0]
            # after deferring, they hit at the first turn >= defer_to with a rank
            hit_turn, rank = None, None
            for t in range(defer_to, MAX_TURNS + 1):
                if len(s["turn_ranks"]) >= t and s["turn_ranks"][t - 1] is not None and s["eligible"][t - 1]:
                    hit_turn, rank = t, s["turn_ranks"][t - 1]
                    break
            if hit_turn is None:
                delta += -(0.5 + 0.3 * (1.0 / r1) + 0.02 * (11 - 1))  # becomes a miss
                dropped += 1
            else:
                delta += 0.3 * (1.0 / rank - 1.0 / r1) - 0.02 * (hit_turn - 1)
        out["ev_delta_if_defer_turn1_cohort"][f"defer_to_turn_{defer_to}"] = {
            "technical_score_delta": round(delta / n_total, 5),
            "sessions_that_become_misses": dropped,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-stopping shadow evaluator (rank-vs-turn curves)")
    parser.add_argument("--agent", default="starter.agent", help="module path exporting Agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results_shadow.json")
    args = parser.parse_args()
    agent_cls = importlib.import_module(args.agent).Agent
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    sessions = shadow_evaluate(agent_cls(args.catalog), samples, catalog_ids, categories, products)
    summary = analyze(sessions)
    Path(args.output).write_text(
        json.dumps({"summary": summary, "sessions": sessions}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
