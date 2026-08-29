# Conversational Search Agent — Method Report

A deterministic constraint state machine that scores **0.962 TechnicalScore** on the public set
with **HitRate@10 = 1.000 on every scenario**, zero LLM calls, zero tokens, and a stdlib-only
runtime. Entry point: `from submission.agent import Agent`.

Two reference points, because they are often conflated: the organizer's shipped weak BM25 starter
scores **0.10671** (`docs/baseline_results.json`, reproduced exactly). Our own strengthened BM25
reference — `starter/agent.py`, which the challenge explicitly invites participants to edit, and
which is *not* the organizer's baseline — scores **0.858944**. Comparisons in this report are
against the strengthened reference, which is the harder of the two.

## 1. Method

The evaluator builds each session's hidden intent card from the target product's own metadata, so
every revealed constraint is a verbatim substring of the target's catalog text, and the opening
message carries the target's exact category tail. The agent is built around those two facts:

- **Shared normal form + exact-substring matching.** One normalization (lowercase; punctuation runs
  → single space; `$ % .` preserved only inside numbers) applied identically to product text and
  constraints, then plain substring matching. Handles the evaluator's `"key: value"` detail
  flattening, tail-side 180-char truncation, and the sentence-final period on the last constraint
  of every reply.
- **Category-anchor retrieval.** Each product's coarse category (the evaluator's own construction)
  is precomputed; the opening message is resolved to an anchor set by a token-*suffix* scan, so any
  rewording of the lead-in verb still lands ("Help me track down {cat}" → same anchor). On the
  public evaluator the target is guaranteed inside the anchor set from turn 1. Applied as a
  dominant boost, never a hard filter; BM25 + loose-term retrieval remain as fallback.
- **Specificity-weighted rerank.** Constraint weight grows with token count (a 120-char feature
  string dominates; a bare material word is a whisper); all-constraints-matched bonus; budget
  constraints are never substring-matched — a permissive numeric regex extracts the amount and
  scores against price only (the card's "budget around $X" is the target's exact price).
- **Always-`other` question policy.** Provable from the simulator code: disclosure is tracked
  per-constraint and `other` matches any constraint type, so repeated `other` asks weakly dominate
  every specific ask. Ask `other` until the card drains, then stop asking.
- **Uninformed-turn recommendation gating** (the single largest lever, +0.066): the session ends at
  the first hit, so an early top-10 shown before constraints exist freezes a popularity-order rank.
  Policy: return depth 1 until the ranked leader matches ≥ 2 active constraints and turn ≥ 3; full
  top-k on a drained card, on dissatisfaction, or from turn 5. Depth 1 keeps genuine rank-1 hits
  (optimal) while blocking rank-2+ lock-ins.
- **Override recovery via demoted evidence.** On "ignore my earlier preference", old constraints are
  demoted to 0.1× weight with an asymmetric rule — matched demoted evidence adds its small weight,
  unmatched costs nothing. That asymmetry *is* the consistency gate: the old preference is still
  true of the target on this evaluator, and if it ever isn't, the boost self-disables.

**Model choice: none.** The plan reserved an LLM extraction layer as a robustness fallback; it was
never needed — the deterministic extractor holds 1.000 hit-rate under our paraphrase stress harness
(§3), so the LLM stage was dropped. The agent runs fully offline by construction.

## 2. Results

Public set, 200 sessions. Score = 0.5·HitRate@10 + 0.3·MRR + 0.2·efficiency.

| Configuration | TechnicalScore | HitRate@10 | MRR | MTTC |
|---|---|---|---|---|
| organizer weak BM25 starter (`docs/baseline_results.json`) | 0.106710 | 0.125 | 0.068034 | 9.81 |
| strengthened BM25 reference (`starter/agent.py`) | 0.858944 | 0.975 | 0.709812 | 3.075 |
| Stage 1: exact-constraint rerank + anchor | 0.895986 | 1.000 | 0.689954 | 1.55 |
| Stage 2: policy + override + robust extraction | 0.895986 | 1.000 | 0.689954 | 1.55 |
| **Stage 3: + recommendation gating (final)** | **0.962179** | **1.000** | **0.946929** | 2.095 |

Stage 1's MRR *drop* is the diagnosis, not a regression: the anchor guarantees the target is in the
pool from turn 1, so 140/200 sessions hit immediately at popularity-order ranks before any
constraint can rerank — verified as a weight-invariant ceiling, fixable only by gating. Stage 2 is
byte-identical on the clean set by design; its value shows under paraphrase.

Final per-scenario:

| Scenario | n | HitRate@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 1.000 | 0.953869 | 1.63 |
| browsing | 80 | 1.000 | 0.947917 | 1.89 |
| boundary | 10 | 1.000 | 0.860 | 2.60 |
| intent_override | 30 | 1.000 | 0.954762 | 3.73 (baseline 5.3) |

Under the paraphrase stress harness (§3):

| Configuration | clean | L1 (frames reworded) | L2 (payloads mutated) |
|---|---|---|---|
| strengthened BM25 reference | 0.858944 | 0.663890 (hit 0.775) | 0.616357 (hit 0.710) |
| Stage 2 | 0.895986 | 0.893538 (hit 1.000) | 0.892211 (hit 1.000) |
| **Stage 3 (final)** | **0.962179** | **0.961979 (hit 1.000)** | **0.959033 (hit 1.000)** |

## 3. Measurement instruments (first-class contributions)

**Non-stopping shadow evaluator** (`tools/shadow_evaluator.py`): replays every session without
stopping at the first hit, recording the target's rank at every turn — the true counterfactual
behind any "defer recommendations" idea. It first *demoted* gating (under the BM25 reference's reranker
the oracle ceiling was +0.004–0.009, mostly eaten by turn cost), then *justified* it after Stage 1
(ranks plateau at ~1 by turn 3 under the anchor reranker; uniform defer-to-turn-2 alone worth
+0.039). Same instrument, opposite verdicts, both correct — the decision procedure working as
designed.

**Paraphrase stress harness** (`tools/paraphrase_eval.py`): the real evaluator loop with every
customer message rewritten by a deterministic seeded paraphraser. L1 rewrites sentence frames but
keeps constraint payloads byte-identical; L2 additionally mutates payloads (joiner replacement,
case flips, budget rephrasing, reordering). An offline cache hook (`data/paraphrase_cache.jsonl`)
allows LLM-generated rewrites to be replayed with no network. The BM25 reference loses 19.5–24.3 points
under it; the final agent loses 0.02–0.31.

**L3 semantic-payload harness** (`tools/l3_paraphraser.py`, `tools/l3_eval.py`). L1 and L2 cannot
support a robustness claim about *lexical matching*, and saying so precisely matters more than the
0.959 headline: `_mutate_payload` returns `cat`, `attr` **and `old`** unchanged — so the
`intent_override` opening's `old_value`, which is a real constraint (`soft[-1]`), is never touched —
and `_mutate_constraint` does exactly two things, both no-ops against this agent by construction
(budget rephrasing, bypassed because budget is regex-extracted and never substring-matched; a
first-letter flip, erased by the lowercasing normal form). L2 tests frame parsing and payload
splitting. It does not test payload matching at all.

L3 rewrites payloads *semantically* — meaning-preserving, so a human reading them identifies the
same product — using rules grounded in the actual public-set constraint distribution (800
instances, 342 distinct, dominated by templated Amazon shapes: `<X> sole`, `<X> closure`,
`N% <Material>`, `Machine Wash`, `color: <c>`). 60% of constraints mutate.

```
"Rubber sole"                → "soles made of Rubber"
"Pull On closure"            → "pull on-style fastening"
"Machine Wash; color: black" → "can go in the washing machine; black in colour"
"95% Polyester, 5% Spandex"  → "Polyester (95 percent) blended with Spandex (5 percent)"
```

**L3a** leaves the category tail byte-identical so the anchor still resolves — the realistic
"organizer added a paraphraser" scenario. **L3b** rewrites the tail too, so the anchor scan fails as
well; it is a floor, not a forecast. Single-word constraints (276 of 800, 34.5%: `cotton`,
`polyester`, `Imported`) are left alone by default — any faithful rewording keeps the head noun, and
they are the floor under the degradation; `--mutate-singles` measures the harsher variant. The
harness reproduces clean and L2 byte-exactly, which is its own correctness check.

## 4. Ablations and cut features

Thresholds tuned on a stratified 151-session tune split; the 49-session validate split used only as
an aggregate guardrail (no-gating 0.894461 → gated **0.964198**, hit-rate 1.000, no loss). Marginal
deltas on the full 200:

| Feature | Score delta | Decision |
|---|---|---|
| recommendation gating | **+0.0662** | ship |
| p_buy gating escape (full lists early for confident buyers) | −0.0027 | cut — re-creates the lock-ins gating exists to prevent |
| profile-tag prior | MRR +0.0025, net −0.00135 (MTTC cost) | cut on net score |
| MMR diversification | exactly 0 (firing condition barely occurs) | cut — pure private-set downside risk |
| stale-rec penalty (dissatisfaction-only) | 0 on clean | ship as failure insurance |
| relaxation ladder (dissatisfaction-only) | 0 on clean | ship as failure insurance |

Cut features remain in the code behind `FEATURE_*` flags (off) with their measured deltas recorded.

### The two signals swap roles under paraphrase

Ablating the anchor *properly* — emptying `_anchor_index`, which removes it from candidate
generation as well as from the score; zeroing `ANCHOR_BONUS` alone under-ablates:

| variant | clean | L3a (3 seeds) |
|---|---|---|
| full agent | 0.962179 (hit 1.000) | 0.887002 (hit 0.963) |
| anchor fully removed | 0.945056 (hit 0.995) | 0.839783 (hit 0.933) |
| | **−0.0171** | **−0.0472** |

On the clean set the anchor is close to redundant — the substring reranker carries almost
everything, though not quite all of it: hit-rate drops 1.000 → 0.995, so one session depends on the
anchor outright. Under paraphrase, substring matching degrades and the anchor becomes the main
carrier, **2.8× more load-bearing than on clean**. That role swap is the defence-in-depth the design
argues for, stated as a number rather than asserted.

### Gating's value is a slope, not a floor

Recommendation gating is the single largest lever and the feature the submission rests on, so its
behaviour under divergence matters more than its clean delta:

| severity | gated | no gating | gating delta |
|---|---|---|---|
| clean | 0.962179 | 0.895986 | **+0.066193** |
| L1 | 0.961979 | 0.893538 | +0.068441 |
| L2 | 0.959033 | 0.892211 | +0.066822 |
| L3a | 0.887002 (hit 0.963) | 0.852811 (hit 0.970) | **+0.034191** |

Flat through L1 and L2 — neither touches payloads — then halving at L3a. The mechanism is visible
in the hit-rate column: **removing gating raises hit-rate under paraphrase** (0.963 → 0.970).
Gating buys rank by spending turns; when the ranking never becomes informed, that spend converts
into pure hit-rate risk. The gate cannot tell, because its own `informed` test counts *exact*
matches and so fails in the same correlated way the reranker does — which is what `REGIME_ESCAPE`
(§4.1) exists to break.

## 5. Cost, latency, tokens

| Metric | Value |
|---|---|
| LLM calls / tokens / API cost | 0 / 0 / $0 |
| One-time index build (50k products) | 4.1 s |
| Full 200-session evaluation | 8.7 s |
| Per session / per turn | 44 ms / 21 ms |
| Runtime dependencies | Python 3 stdlib only |

## 6. Limitations

1. **Semantic payload rewriting — measured, and it was the real exposure.** L1/L2 showed near-zero
   degradation, but that reflected shared assumptions, not robustness: L2's payload mutations are
   no-ops against this agent by construction (§3). The L3 harness rewrites payloads semantically,
   and the Stage 3 agent degrades materially under it:

   | | clean | L3a (payloads rewritten) | L3b (category tail too) |
   |---|---|---|---|
   | Stage 3 | 0.962179 (hit 1.000) | 0.887002 (hit 0.963) | 0.832112 (hit 0.925) |
   | Stage 4 | 0.963479 (hit 1.000) | **0.940572 (hit 0.993)** | **0.871357 (hit 0.935)** |

   The named mechanism is that `constraint.norm in text` is all-or-nothing, so a payload reworded
   anywhere scores exactly as badly as an unrelated product; the constraint then subtracts
   `UNMATCHED_PENALTY` from every candidate including the target, `ALL_MATCHED_BONUS` never fires,
   and ranking inside the anchor collapses to popularity order. §4.1 addresses that mechanism
   directly. Dense retrieval was still not needed: the residual is lexical, and graded lexical
   matching closes most of it at zero clean cost.

   **The caveat survives the fix.** The L3 rewriter shares authorship with the agent exactly as
   L1/L2 did. These are measurements on our own instrument, not a guarantee about an organizer's.
   What can be claimed without the instrument is narrower and still useful: every change in §4.1
   replaces a *binary* decision that silently discards information with a graded one, and each is
   clean-neutral or better by construction, so none of them trades public-set score for a
   speculative private-set gain.
2. **The core signals assume spec-faithful card construction.** Exact-substring and anchor
   dominance rely on the evaluator building intent cards from target metadata (spec-guaranteed).
   A differently generated private set weakens them; the BM25 and loose-term fallbacks then carry.
3. **Validate split is 49 sessions** — an aggregate guardrail only. Per-scenario numbers are
   reported on the full 200 and carry the corresponding overfit risk.
4. **Gating trades speed for rank by design:** MTTC rose 1.55 → 2.095. At 0.02 score per turn vs
   0.3 per unit of reciprocal rank, the trade is strongly positive, but MTTC-sensitive judges
   should note it is deliberate.
5. **The clean set is nearly saturated, and the remainder is mostly unreachable.** Hit-rate is
   exhausted at 1.000 and MTTC is at its policy floor, not near it (override sessions cannot
   convert before the evaluator fires the override on turn 3 or 4: floor 3.667 against 3.733
   measured; boundary sessions lose turn 2 to the one-off by construction). That leaves ~0.0163,
   almost all of it MRR, and a practical ceiling near **0.9785**. Replaying the 17 sub-rank-1
   sessions through the shadow evaluator shows **15 have flat rank trajectories across all ten
   turns** — every card constraint revealed, rank unmoved. They are ties among products that match
   every constraint (`public_0083`: 31 of 681 anchor-set products match all four of `polyester`,
   `100% Polyester`, `Imported`, `Button closure`), broken by the popularity prior. No policy
   change reaches them, and tuning for the two that would improve is overfitting to two sessions.
   Effort therefore went to robustness, where the measured headroom was ~9× larger. Full working:
   `docs/headroom_and_robustness.md`.
6. **The budget path is entirely untested by every number in this report.** `intent_card` appends
   `budget around $X` after the material/colour inserts and the features/details entries, so it
   falls outside `cleaned[:4]` for all 200 public targets: **zero budget constraints are ever
   revealed**. The same generator makes it equally unreachable on the private set, so the risk is
   dead weight rather than error — but `BUDGET_*` weights and `_budget_score` have never executed
   on real data. The misrouting guard in §4.1 exists because the *classifier* in front of that
   path did fire, on ordinary features text.

## 7. Reproduction

Python 3.12 tested; no third-party packages required (`requirements.txt`). Everything runs offline.

```bash
python3 -m unittest discover -s tests                                  # 53 tests
python3 -m tools.run_eval --agent submission --output results.json     # full evaluation
python3 -m tools.paraphrase_eval --agent submission.agent --level L2   # stress harness
python3 -m tools.shadow_evaluator --agent submission.agent             # rank-vs-turn curves
python3 -m tools.demo_session --sample-id public_0003                  # the transcript below
```

The official harness needs only `submission/agent.py` (+ `submission/__init__.py`):
`from submission.agent import Agent`, then `reset(...)` / `respond(...)` per the contract.

## 8. Demo session (intent_override, verbatim replay)

```text
session public_0003  scenario=intent_override
target  B09YMTWDXJ  Casio Men's Wrist Watch AQ-800E-7A  [$47.95]
card    hard=['Water Resistant', '3 Year Battery']  soft=['Day / Date Indicator', 'Stainless Steel Band']
==============================================================================

Turn 1
  customer > I'm looking for Watches Wrist Watches. Stainless Steel Band
  agent    > Here are my strongest matches so far. Is there another must-have
             detail I should factor in?  (ask_attribute='other')
       1. Invicta Men's Pro Diver Quartz Diving Watch with Stainless-Steel Stra...  [$69.99]

Turn 2
  customer > For that, what matters is: Water Resistant; 3 Year Battery.
  agent    > Here are my strongest matches so far. Is there another must-have
             detail I should factor in?  (ask_attribute='other')
       1. Casio Men's Wrist Watch AQ-800E-7A  [$47.95]  <-- TARGET

Turn 3
  customer > Actually, ignore my earlier preference. What I need is: Water Resistant.
  agent    > Here are my strongest matches so far. Is there another must-have
             detail I should factor in?  (ask_attribute='other')
       1. Casio Men's Wrist Watch AQ-800E-7A  [$47.95]  <-- TARGET

HIT at turn 3, rank 1
```

Everything the design argues for is visible here: turns 1–2 are gated to depth 1 while the ranking
is uninformed; the constraints revealed by the `other` ask put the target at rank 1 on turn 2 —
which the evaluator correctly ignores, because in override sessions hits don't count before the
override fires; the override arrives at turn 3 and converts at rank 1. The turn-2 non-hit is the
exact mechanism (evaluator's `override_applied` gate) that makes an unconditional stale-rec penalty
dangerous — and why ours fires only on explicit dissatisfaction.
