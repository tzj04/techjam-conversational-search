"""Run the real evaluator loop against a generated catalog and generated sessions.

Local analysis tool only — never part of the submission bundle.

Why this exists: `data/catalog.jsonl` is a 50,000-row download, not repository
content, so on a fresh checkout *every* end-to-end instrument in this repo is
unavailable — `run_eval`, `paraphrase_eval`, `l3_eval` and `shadow_evaluator`
all require it. The unit tests exercise a three-product fixture and the report's
figures are the only end-to-end evidence. That leaves a change with no way to
show it did not break the conversation loop.

This builds a catalog with the same *shape* as the real one (the templated
Amazon metadata that `docs/headroom_and_robustness.md` §5 documents: `N% <M>`,
`<X> closure`, `<X> sole`, `Imported`, a `details` dict, a price, a rating) and
samples sessions from it in the released scenario mix, then hands both to the
shipped evaluator. Intent cards, the override schedule and the boundary one-off
are produced by `evaluator/local_evaluator.py` itself, unmodified.

**What it is and is not.** It is a regression detector: `--against` replays two
agent modules over identical sessions and diffs the per-session rank and hit
turn, so a change that alters behaviour anywhere in the loop is visible without
the download. It is *not* a headroom measurement — a generated catalog has a far
smaller vocabulary than 50k real products, so constraint matching alone
identifies the target and the absolute scores are not comparable to the report's.

    python3 -m tools.synthetic_eval --seeds 0,1,2
    python3 -m tools.synthetic_eval --level catdrift --seeds 0,1,2
    python3 -m tools.synthetic_eval --against baseline.agent --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import importlib
import json
import random
import statistics
import tempfile
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate
from tools.l3_paraphraser import L3Paraphraser
from tools.paraphrase_eval import paraphrase_evaluate
from tools.paraphraser import Paraphraser

MATERIALS = ["Cotton", "Polyester", "Leather", "Wool", "Nylon", "Silk", "Rayon", "Spandex"]
COLORS = ["black", "white", "blue", "red", "pink", "green", "brown", "gray"]
CATEGORIES = [
    ("Women", "Dresses"), ("Men", "Boots"), ("Women", "Sports Bras"),
    ("Girls", "Sandals"), ("Men", "Shirts"), ("Women", "Handbags"),
    ("Boys", "Sneakers"), ("Women", "Necklaces"), ("Men", "Watches"),
]
CLOSURES = ["Button", "Zipper", "Pull On", "Lace-up", "Hook and Loop"]
SOLES = ["Rubber", "Synthetic", "Leather", "EVA"]
ORIGINS = ["Imported", "Made in the USA", "Made in the USA or Imported"]
CARE = ["Machine Wash", "Hand Wash Only"]
# Real Amazon features text cross-references other categories ("A great match
# for Women Dresses"). That matters here because the anchor scan reads verbatim
# catalog text whenever no anchor is held.
XREF_RATE = 0.35
SCENARIO_MIX = ["buying"] * 4 + ["browsing"] * 4 + ["intent_override"] * 3 + ["boundary"]


def build_catalog(seed: int, count: int) -> list[dict]:
    rng = random.Random(f"catalog\0{seed}")
    rows: list[dict] = []
    for index in range(count):
        group, leaf = rng.choice(CATEGORIES)
        material = rng.choice(MATERIALS)
        features = [
            f"{rng.randint(50, 100)}% {material}",
            f"{rng.choice(CLOSURES)} closure",
            rng.choice(ORIGINS),
        ]
        if rng.random() < XREF_RATE:
            features.append(f"A great match for {' '.join(rng.choice(CATEGORIES))}")
        rows.append({
            "parent_asin": f"B{index:05d}",
            "title": f"{rng.choice(COLORS).title()} {material} {leaf[:-1]} Model {index}",
            "categories": ["Clothing, Shoes & Jewelry", group, leaf],
            "features": features,
            "details": {
                "Sole material": rng.choice(SOLES),
                "Department": group,
                "Care instructions": rng.choice(CARE),
            },
            "description": [f"A {rng.choice(COLORS)} {leaf[:-1].lower()} for everyday wear."],
            "store": f"Store{rng.randint(1, 40)}",
            "price": round(rng.uniform(9.0, 220.0), 2),
            "average_rating": round(rng.uniform(3.0, 5.0), 1),
            "rating_number": rng.randint(1, 20000),
        })
    return rows


def build_sessions(rows: list[dict], seed: int, count: int) -> list[dict]:
    rng = random.Random(f"sessions\0{seed}")
    targets = rng.sample(rows, min(count, len(rows)))
    return [{
        "sample_id": f"synth_{index:04d}",
        "scenario_type": SCENARIO_MIX[index % len(SCENARIO_MIX)],
        "ground_truth": {"parent_asin": row["parent_asin"]},
        "user_profile": {
            "preference_tags": ["fit", "comfort", "durability"],
            "summary": "Prior purchases emphasize fit, comfort, durability.",
        },
    } for index, row in enumerate(targets)]


def run_one(module_path: str, level: str, seed: int, products: int, samples: int) -> dict:
    rows = build_catalog(seed, products)
    path = Path(tempfile.mkdtemp()) / "catalog.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    session_specs = build_sessions(rows, seed, samples)
    catalog_ids, categories, catalog = catalog_index(path)
    agent = importlib.import_module(module_path).Agent(path)
    if level == "clean":
        return evaluate(agent, session_specs, catalog_ids, categories, catalog)
    if level in ("L1", "L2"):
        paraphraser = Paraphraser(level=level, seed=seed)
    else:
        paraphraser = L3Paraphraser(level=level, seed=seed)
    return paraphrase_evaluate(agent, session_specs, catalog_ids, categories, catalog, paraphraser)


def digest(result: dict) -> list[str]:
    """Per-session outcome, for an exact before/after diff."""
    return [
        f"{session['sample_id']}:{session['best_rank']}:{session['first_hit_turn']}"
        for session in result["sessions"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="synthetic-catalog evaluation harness")
    parser.add_argument("--agent", default="submission.agent")
    parser.add_argument("--against", default=None,
                        help="second agent module; diffs per-session outcomes")
    parser.add_argument("--level", default="clean",
                        choices=["clean", "L1", "L2", "L3a", "L3b", "catdrift"])
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--products", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=120)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    scores: list[float] = []
    changed_total = 0
    for seed in seeds:
        result = run_one(args.agent, args.level, seed, args.products, args.samples)
        scores.append(result["recommended_technical_score"])
        line = (f"seed {seed}: score={result['recommended_technical_score']:.6f} "
                f"hit={result['hit_rate_at_10']:.3f} mrr={result['mrr']:.4f} "
                f"mttc={result['mttc']:.3f}")
        if args.against:
            other = run_one(args.against, args.level, seed, args.products, args.samples)
            changed = [
                (a, b) for a, b in zip(digest(other), digest(result)) if a != b
            ]
            changed_total += len(changed)
            line += (f"  | {args.against} score={other['recommended_technical_score']:.6f}"
                     f"  delta={result['recommended_technical_score'] - other['recommended_technical_score']:+.6f}"
                     f"  sessions changed={len(changed)}")
            if changed:
                line += f"\n    first diffs (against -> agent): {changed[:5]}"
        print(line)

    if len(seeds) > 1:
        print(f"\nmean={statistics.fmean(scores):.6f} "
              f"+/- {statistics.stdev(scores):.6f}")
    if args.against:
        print(f"total sessions changed vs {args.against}: {changed_total}"
              f" / {len(seeds) * args.samples}")


if __name__ == "__main__":
    main()
