"""L3 stress harness: the real evaluator loop with *semantically* rewritten
constraint payloads (see tools/l3_paraphraser.py for why L2 does not test this).

Local analysis tool only — never part of the submission bundle.

    python3 -m tools.l3_eval --agent submission.agent --level L3a
    python3 -m tools.l3_eval --agent submission.agent --level L3a --seeds 0,1,2
    python3 -m tools.l3_eval --agent submission.agent --level L3a --ablate anchor
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl
from tools.l3_paraphraser import L3Paraphraser
from tools.paraphrase_eval import paraphrase_evaluate
from tools.paraphraser import Paraphraser


def build_agent(module_path: str, catalog: str, ablate: str | None):
    agent = importlib.import_module(module_path).Agent(catalog)
    if ablate == "anchor":
        # Ablate the anchor *properly*: emptying the index removes it from
        # candidate generation too (agent.py `_build_pool`), not just from the
        # score. Zeroing ANCHOR_BONUS alone under-ablates.
        agent._anchor_index = {}
    elif ablate == "gating":
        agent_module = importlib.import_module(module_path)
        agent_module.FEATURE_GATING = False
    elif ablate not in (None, "none"):
        raise SystemExit(f"unknown ablation: {ablate}")
    return agent


def run_one(module_path: str, level: str, seed: int, catalog: str, samples,
            catalog_ids, categories, products, ablate: str | None,
            mutate_singles: bool) -> dict:
    if level == "clean":
        paraphraser = Paraphraser(level="none", seed=seed)
    elif level in ("L1", "L2"):
        paraphraser = Paraphraser(level=level, seed=seed)
    else:
        paraphraser = L3Paraphraser(level=level, seed=seed, mutate_singles=mutate_singles)
    agent = build_agent(module_path, catalog, ablate)
    return paraphrase_evaluate(agent, samples, catalog_ids, categories, products, paraphraser)


def main() -> None:
    parser = argparse.ArgumentParser(description="L3 semantic-paraphrase stress harness")
    parser.add_argument("--agent", default="submission.agent")
    parser.add_argument("--level", default="L3a",
                        choices=["clean", "L1", "L2", "L3a", "L3b"])
    parser.add_argument("--seeds", default="0", help="comma-separated seeds")
    parser.add_argument("--ablate", default=None, choices=["anchor", "gating", "none"])
    parser.add_argument("--mutate-singles", action="store_true",
                        help="also reword single-word constraints (harsher than default)")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]

    runs = []
    for seed in seeds:
        result = run_one(args.agent, args.level, seed, args.catalog, samples,
                         catalog_ids, categories, products, args.ablate,
                         args.mutate_singles)
        runs.append(result)
        print(f"seed {seed}: score={result['recommended_technical_score']:.6f} "
              f"hit={result['hit_rate_at_10']:.3f} mrr={result['mrr']:.4f} "
              f"mttc={result['mttc']:.3f}")

    scores = [run["recommended_technical_score"] for run in runs]
    summary = {
        "agent": args.agent, "level": args.level, "seeds": seeds,
        "ablate": args.ablate, "mutate_singles": args.mutate_singles,
        "score_mean": round(statistics.fmean(scores), 6),
        "score_stdev": round(statistics.stdev(scores), 6) if len(scores) > 1 else 0.0,
        "hit_mean": round(statistics.fmean(run["hit_rate_at_10"] for run in runs), 6),
        "mrr_mean": round(statistics.fmean(run["mrr"] for run in runs), 6),
        "mttc_mean": round(statistics.fmean(run["mttc"] for run in runs), 6),
        "runs": [{key: value for key, value in run.items() if key != "sessions"} for run in runs],
    }
    if len(scores) > 1:
        print(f"\nmean={summary['score_mean']:.6f} +/- {summary['score_stdev']:.6f} "
              f"hit={summary['hit_mean']:.4f}")
    if args.output:
        payload = {**summary, "sessions": runs[0]["sessions"]}
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
