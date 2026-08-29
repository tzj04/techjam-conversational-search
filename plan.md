# Plan: Conversational Search Agent

## TL;DR

Build a **slot-based constraint state machine** with a **cascading retriever** (exact-phrase → BM25 → dense embeddings), an **information-gain question policy**, and **soft intent routing**: a continuous buyer-confidence scalar that blends constraint-driven scoring with profile/popularity priors, instead of a hard buyer/browser branch. An LLM is used only for robust intent/slot extraction and message phrasing (with a deterministic regex fallback so the agent runs fully offline). No attention-mechanism machinery — but the *concepts* behind that idea (weighted evidence, recency dominance, confidence-blended recommendations) survive as three scalar weights in the reranker; see §2.1.

**Measured baseline (current `starter/agent.py`, public set, 2026-08-29):** TechnicalScore **0.859**, HitRate@10 **0.975**, MRR **0.710**, MTTC **3.08**. Per scenario: buying 0.99 hit / MTTC 2.6, browsing 0.975 / 2.7, boundary 1.0 / 2.6, intent_override **0.933 / 5.3**. (Run: `python3 -m evaluator.local_evaluator` — defaults point at `data/catalog.jsonl` / `data/public_set.jsonl`. Ignore the weak_bm25 0.107 reference in `docs/baseline_results.json`.)

> **Final outcome (2026-08-29, all stages complete — see `REPORT.md`):** TechnicalScore **0.962** (HitRate@10 **1.000** on every scenario, MRR **0.947**, MTTC 2.10), holding **0.962 / 0.959** under the L1/L2 paraphrase stress harness where the starter fell to 0.664/0.616. Deterministic, stdlib-only, zero tokens. Stage-by-stage outcomes are noted inline below.

The three levers, in order of points available:

1. **MRR 0.71 → ~0.9** (30% of score). Two causes, both measured in `results_starter.json`:
   - **Rank lock-in:** 71 of 195 hits happen on **turn 1** at mean rank **2.8** — the session ends at first hit, freezing a mediocre rank before any constraint was revealed. The non-stopping shadow run (`tools/shadow_evaluator.py`, 2026-08-29) confirms the counterfactual directly: that same cohort's mean rank improves to **1.81 by turn 3** and **1.35 by turn 4** if the session continues, with zero sessions leaving the top-10. Main fix: exact-constraint reranking (§6 Stage 1), which lifts ranks at *every* turn; recommendation gating (§5.4) started as a small measured bonus under the starter ranking (+0.004–0.009 ceiling) and — exactly as §5.4's re-measure-then-decide procedure was designed to catch — became the **+0.066 headline feature** once the Stage 1 anchor guaranteed the target in the pool from turn 1 (outcome note in §5.4).
   - **Fuzzy matching:** the starter tokenizes revealed constraint strings, destroying the exact-substring signal (§1 fact 1).
2. **Intent-override MTTC 5.3 → ~4** (floor ≈ 3.5, override fires turn 3–4). We currently lose ~1.8 turns re-converging. Fix: §5.5 — including the discovery that the "old" preference is still true of the target (§1 fact 10).
3. **Protect HitRate 0.975 on the private set**, which may paraphrase customer messages — the paraphrase stress harness built early so every stage is measured under it (§6 Stage 1.5), with the BM25 → dense fallback cascade as the fix (§6 Stage 4).

---

## 1. What the evaluator actually does (read this first — it drives every design choice)

From `evaluator/local_evaluator.py` and `docs/competition_specification.md`:

1. **The hidden target's "intent card" is built from its own metadata.** `hard_constraints` / `soft_preferences` are literal (≤180-char) strings from the product's `features`, `details`, detected material, color, and price. When the customer says *"For that, what matters is: 100% cotton; Machine wash"*, those are **exact substrings of the target product's catalog text**. Exact-phrase search on a specific revealed constraint often narrows 50k products to a handful — our strongest signal by far.
2. **Constraints are revealed only in response to `ask_attribute`.** `customer_reply()` returns up to 2 undisclosed constraints whose `classify_constraint()` matches the asked attribute. Asking `other` matches **any** constraint type → highest information gain per turn. Asking `brand` or `category` reveals nothing (the classifier never returns those types).
3. **Returning `ask_attribute: null` before a hit wastes a turn** — the customer replies "ask me about one specific attribute." So: always ask something until the card is drained.
4. **The first message gives the coarse category** — literally the last 2 non-generic entries of the target's `categories` field, joined ("I'm looking for {Women Dresses}..."). Exact category-tail match is a strong hard-ish filter, better than treating them as loose terms. (Verified in the starter: `category_terms` are tokenized into loose terms and scored softly at +1.4/−0.2 each in `_slot_score` — they *do* persist across turns, so the observed cross-turn category drift comes from the soft weight being overwhelmed, not from the terms being dropped. The Stage 1 sticky filter is the right fix.)
5. **Scenario is identifiable from message shape** (see §5.1).
6. **Only the rank of the exact `parent_asin` matters**, and per-session score algebra says rank dominates speed: a hit is worth ~0.5, rank 1 vs rank 3 is worth 0.2, one extra turn costs only 0.02. **A rank-1 conversion is worth ~10 turns of delay.** Optimize rank first, speed second, but never risk the hit itself (a miss forfeits ~0.7).
7. **Exceptions/timeouts = miss.** Everything gets try/except and a deterministic fallback.
8. **The private evaluator may paraphrase customer messages** ("paraphrasing cannot decide correctness" — but it can change surface forms). Every exact-match feature needs a lexical → dense fallback path.
9. **The session ends at the FIRST hit** (`local_evaluator.py:252-255`). Combined with fact 6, this is the rank lock-in problem from the TL;DR: an early top-10 appearance at rank 5 permanently costs 0.24 vs converging to rank 1 two turns later. This motivates deliberately *limiting* early recommendations while uncertainty is high (§5.4).
10. **The intent-override "old" preference is still true of the target.** `behavior_for()` sets `old_value = soft_preferences[-1]` and `new_value = hard_constraints[0]` — *both from the same target's card*. The override message ("ignore my earlier preference") replaces a soft preference with a hard constraint of the *same product*. The old string still exact-matches the target — and it is never marked disclosed, so a later `ask` can even re-reveal it. **Don't discard it: demote it to low-weight evidence** with a consistency gate (§5.5). This is the key to closing the override MTTC gap.
11. **Exact-substring matching needs a shared normal form — naive `in` checks silently fail.** The evaluator flattens `details` dicts as `"key: value"` but the agent's product text joins them as `"key value"` (no colon); internal whitespace isn't collapsed on the product side; constraints are truncated at 180 chars (possibly mid-word); reply constraints are joined with `"; "` and suffixed `"."`. Implementation: normalize both sides (lowercase, collapse all punctuation runs to single spaces, keep `$ % .` inside numbers), split replies **only** on `"; "`, **strip the sentence-final `"."` from the last segment** — card constraints never end in punctuation (`_clean_constraint` strips `" -;,."` at build time), so the trailing period is pure sentence suffix and would otherwise silently break substring matching for the final constraint of *every* reply. Strip the `"color: "` generated prefix; extract budget by regex, not prefix (fact 12). Truncation is tail-side (verified: `[:180].rstrip()`), so plain substring (not word-boundary) matching handles a truncated final partial word — micro-tests for both edge cases in §8.
12. **Two more literal signals hiding in the card generator:** `"budget around $X"` means X **is the target's exact price** — treat as near-equality (±cents), with a ±35% range fallback for private-set safety. Extract the number with a permissive regex over the whole constraint (`\$?\d[\d,]*\.?\d*` near budget-ish words), **never** by stripping the literal `"budget around $"` prefix — a paraphraser rendering "my budget's about $80" defeats prefix-stripping; if no number extracts, apply **no** price filter rather than a wrong one. And a bare material word as the first constraint means the material regex matched the target's corpus — single-word constraints match thousands of products, so **weight each constraint by its specificity** (length / token count), not uniformly.
13. **In intent_override sessions, hits before the override turn don't count.** The hit check is `if override_applied and target in ranked` (`local_evaluator.py:252`) — on turns 1–2 of an override session the target can sit at rank 1 in our returned list and the session still continues. Two consequences: (a) "shown but no hit" does **not** prove a product isn't the target — an unconditional penalty on previously-shown items would push the target down in exactly the 30 override sessions (§5.5); (b) early rank in override sessions is worthless, so withholding recommendations there on turns 1–2 is free (§5.4).

## 2. Verdicts on our earlier ideas

| Idea | Verdict |
|---|---|
| Embed items into a vector field | **Allowed and worth doing — as insurance.** Spec lists "keyword, dense, or hybrid retrieval" as in scope; what's banned is *infrastructure-heavy vector databases*. A precomputed `.npy` matrix + brute-force numpy cosine (50k × 384 float16 ≈ 38 MB) is a "lightweight local asset". Its job is robustness against private-set paraphrase, not the core signal. |
| Hybrid attention / CSA / HSA / sliding-window to condense history | **Drop the machinery, keep the concepts.** Those are transformer internals for long-context models. Our "history" is ≤10 short messages and each session is `reset()` fresh — there is nothing to compress and nothing to cache across sessions. What the idea was *really after* survives as scalar weights — see §2.1. |
| Time decay on the profile | **Simplify to last-write-wins per slot** + a small rerank bonus for the latest message's terms (the starter already has this at weight 0.18). With ≤10 turns there is no decay curve to model; the only hard recency event is the explicit override, which is a gate, not a decay. |
| User overrides (black → blue shoes) | **Keep — as slot overwrite + demotion, not deletion.** New value replaces the old slot value; on the explicit override message, old evidence is *demoted*, not erased, because on this evaluator it is still true of the target (§1 fact 10). |
| Recommend browsers from search/purchase history | **The data doesn't exist** — verified: `user_profile` is only `{average_prior_rating, preference_tags: ["fit","comfort","durability"], purchase_frequency: "3-4 prior purchases", rating_style, summary}`. No items, no categories, no search terms, per spec ("raw purchase histories have been removed"). What survives: preference tags as a soft rerank prior and as input to explanation strings — which the spec explicitly lists as an innovation direction ("safe personalization using the aggregate profile"). |

### 2.1 Salvaging the concepts: what "attention over history" becomes here

The instinct behind the attention/decay/80-20 ideas is correct — *recommendations should be a weighted mixture of evidence sources, with weights set by recency and by how confident we are in the customer's intent*. The application-level form of each:

| Concept we wanted | What it becomes in this system |
|---|---|
| Attention over conversation history | **Evidence-weighted score mixture.** Each piece of evidence (revealed constraint, category tail, budget, profile tag, popularity) contributes to the rerank score with a weight set by its *specificity* and *provenance* — a verbatim 120-char constraint string outweighs everything; a profile tag is a whisper. That's "attention" done with five scalars instead of a transformer. |
| 80% buyer confidence → 80% targeted recs | **Soft intent routing** (§5.1). A continuous `p_buy` interpolates the reranker between constraint-dominant scoring and prior-dominant scoring, and sets how aggressively the top-10 is diversified. Exactly the weighted-recommendation blend, as one interpolation parameter. |
| Time decay — newer over older | **Last-write-wins slots** + latest-turn term bonus + the override gate. Discrete, exact, and matches how the simulator actually behaves. |
| Cross-session history caching | **Nothing to cache** — sessions are independent (`reset()` per session) and the profile is aggregate-only. The profile prior *is* the entire cross-session signal the organizers left us. |

This mapping goes in the report almost verbatim — it *is* the "structured constraint state, intent override handling, and dynamic context construction" innovation direction, told as a story of choosing the right abstraction level.

## 3. Architecture

One pipeline per `respond()` call:

```
user_message, turn, profile
        │
        ▼
┌─────────────────────┐   LLM extractor (if API key present)
│ 1. NLU / EXTRACTION │──  else deterministic regex extractor
└─────────────────────┘   → {intent signals, slots, verbatim constraints,
        │                    overrides, no_preference}
        ▼
┌─────────────────────┐   slot store: last-write-wins; override demotes (not
│ 2. STATE UPDATE     │   deletes) old evidence; constraints kept VERBATIM;
└─────────────────────┘   update p_buy (intent confidence) + constraint mass
        │
        ▼
┌─────────────────────┐   a) exact substring match on normalized constraints
│ 3. RETRIEVAL        │   b) SQLite FTS5 BM25 on slot terms (already built)
│    (cascade)        │   c) dense embeddings (precomputed .npy, numpy dot)
└─────────────────────┘   union pool ~200 candidates
        │
        ▼
┌─────────────────────┐   score = w(p_buy)·constraint score (specificity-
│ 4. RERANK           │   weighted exact hits, category tail, budget≈price)
└─────────────────────┘   + (1−w)·prior score (profile tags, rating pop.)
        │                 − exclusions (stale-rec penalty: §5.5 path ONLY)
        ▼
┌─────────────────────┐   ask_attribute maximizing expected info gain;
│ 5. QUESTION POLICY  │   top-N sizing by confidence (rec gating, §5.4);
└─────────────────────┘   diversify across unknown attributes
        │
        ▼
{message (+ transparent "because…" explanation), ask_attribute,
 recommendations[≤10], usage}
```

**Where the LLM fits (and where it doesn't):** retrieval and ranking stay deterministic and local (fast, free, offline-safe). The LLM earns its keep only in step 1 — parsing paraphrased/messy text into slots and verbatim constraint spans — and optionally phrasing `message`. Every LLM call has a regex fallback; a `USE_LLM` env flag makes the whole agent run offline (required: organizer may disable network for final scoring). Headline for the report: **the deterministic path alone scores ~0.86+ at zero tokens** — the LLM is a robustness layer, not a dependency.

## 4. State model (replaces "hybrid attention profile")

```python
@dataclass
class SessionState:
    profile: dict                      # aggregate profile → soft prior + explanations
    p_buy: float                       # intent confidence, 0=browsing…1=buying (§5.1)
    category_terms: list[str]          # last-2 category parts from turn 1 — sticky
    slots: dict[str, str | set]        # material/color/size/style/budget/feature/use_case
    revealed_constraints: list[str]    # VERBATIM strings from replies — the gold signal
    demoted_evidence: list[str]        # pre-override constraints, kept at low weight
    exclusions: set[str]               # "not leather", "avoid heels"
    exhausted: set[str]                # answered "no preference" — never ask again
    asked: list[str]                   # attributes already asked
    stale_recs: set[str]               # shown asins; penalized ONLY on dissatisfaction (§5.5, §1 fact 13)
    drained: int                       # consecutive "no additional preference" replies
    budget_min/max/point: float | None # point ≈ exact target price when card-style
```

Rules:

- **Override handling** (black → blue and the scripted override): a new value for a filled slot *replaces* it. The explicit override message ("actually / ignore my earlier preference / instead") moves current soft slots and `revealed_constraints` into `demoted_evidence` (weight ~0.1× — see the consistency gate in §5.5), keeps `category_terms`, **clears `stale_recs`** (pre-override showings prove nothing — §1 fact 13), extracts the new requirement verbatim, and sets `p_buy` high.
- **No-preference** ("I don't have a preference for X; use your judgment") → add X to `exhausted`, clear that slot, never re-ask, do **not** treat the sentence as content (the starter's regex already does this — keep it). "No *additional* preference" also increments `drained`.
- **Revealed constraints are stored verbatim** and matched via the shared normal form of §1 fact 11 — never tokenized.

## 5. Policies

### 5.1 Soft intent routing (buyer/browser as a confidence, not a branch)

Initialize `p_buy` from the turn-1 message shape, then update each turn:

- "A key requirement is: …" → 0.95 (a hard constraint was just disclosed — extract it verbatim).
- "… but I'm still exploring." → 0.15.
- Override message → 0.9 (they now know exactly what they want).
- Anything else / paraphrased → regex first, LLM classifier fallback; default 0.5.
- Each revealed constraint adds confidence proportional to its specificity ("constraint mass").

What `p_buy` modulates — this is the whole mechanism, three dials:

1. **Rerank blend:** `score = w·constraint_score + (1−w)·prior_score`, `w = f(p_buy, constraint_mass)`. Priors = profile-tag overlap + rating popularity. High confidence → constraints behave like filters; low → cast wide, lean on priors.
2. **Diversification strength** of the top-10 (§5.3): browsers get spread, buyers get depth.
3. **Recommendation gating** threshold (§5.4): low confidence → show fewer, ask more.

Honesty note for us: on the *public* set the shapes are regex-separable, so the blend saturates to ~0/1 and acts like the binary route. Its real value is (a) graceful degradation if the private set paraphrases the opening message, and (b) it is the correct implementation of the "80% buyer → 80% targeted" idea. Both belong in the report.

### 5.2 Question policy

Code-verified: `customer_reply` tracks disclosure **per constraint** (`value not in disclosed`), and `other` matches any constraint type — so repeated `other` asks never repeat a constraint and always return a superset of what any specific-attribute ask could reveal. **Asking `other` weakly dominates every specific ask.** Entropy-based attribute selection is therefore *deleted* from the ask policy — it optimizes a question the simulator never poses. (`_attribute_diversity()` survives, but only for §5.3 top-10 diversification — a different computation over a different population.)

1. **Ask `other` every turn until drained.** The card holds ≤4 unique constraints; each `other` reveals up to 2.
2. **Never ask** `brand`/`category` (`classify_constraint` never returns those — guaranteed wasted turn) or attributes in `exhausted`. Mostly moot under rule 1; kept as a guard.
3. **Stop asking** (`ask_attribute: null`) only when `drained ≥ 2` — the card is empty; from then on rely on ranking + diversification. Never return `null` before that (§1 fact 3). Note: `null` without a hit triggers the "not quite right yet" reply — under this policy that is the *designed* §5.5 strategy-switch moment, not an accident.
4. **Sanity A/B in Stage 2:** always-`other` vs the old ladder (`other` turns 1–2, then highest-diversity attribute) on the public set; the ablation row documents the winner either way (expected: tie or always-`other`, per the dominance argument).
5. **Demo-only guard:** a message with no extractable category, slots, or constraints (a human typing "Hi" into `chat.py`) → empty recommendations + an `ask_attribute`, so interactive demos never fuzzy-match nonsense ("HI-TEC" boots). One line; the evaluator never produces such a message.

### 5.3 Top-10 construction

Rank by target-probability, then **diversify across residual uncertainty** instead of returning 10 near-duplicates:

- Constraint-satisfying candidates first, sorted by blended score.
- For attributes we *don't* know (unfilled or "no preference"), spread slots across values (unknown color → don't return 10 black shoes). Cheap MMR-style pass: penalize a candidate sharing all unknown-attribute values with an already-picked one. Diversification strength scales with `1 − p_buy` and pool entropy.
- Dedupe `parent_asin`, catalog-valid ids only, at most `top_k`.

### 5.4 Early-turn recommendation gating

> **Outcome (2026-08-29, Stage 3):** the decision procedure below fired in the opposite direction from its starter-era estimate — under the Stage 1/2 agent, whose category anchor puts the target in the pool from turn 1, 140/200 sessions hit on turn 1 before any constraint could rank them, and the shadow re-run (`results_shadow_stage2.json`) showed ranks plateauing at ~1 by turn 3. Shipped policy: **depth 1 until the ranked leader matches ≥ 2 active constraints and turn ≥ 3; full top-k on `drained ≥ 1`, dissatisfaction, or turn ≥ 5.** Worth **+0.0662 TechnicalScore** (0.896 → 0.962); validate-split guardrail passed (0.9642, hit-rate 1.0). The starter-era numbers below are kept as the record of why gating was *initially* demoted — both measurements are the same §5.4 procedure doing its job.

Originally justified by comparing turn-1 hits (mean rank 2.8) with turn-≥3 hits (mean rank 2.0) — a confounded comparison (different session populations). Replaced with the real counterfactual: `tools/shadow_evaluator.py` re-runs every session **without stopping at the first hit** and records the target's rank in the returned top-10 at every turn (`results_shadow.json`, 2026-08-29, starter agent):

- For the 71 turn-1-hit sessions, mean rank by turn: **2.77 → 2.34 → 1.81 → 1.35 → 1.31** (turns 1–5). The rank improvement is real, not a selection artifact — the confound actually ran the other way (true turn-3 counterfactual 1.81 beats the confounded 1.98 estimate).
- Zero of the 71 sessions become misses under deferral up to turn 4 (target stays in/returns to the top-10).
- Exact TechnicalScore delta with oracle targeting of this cohort: **+0.0004 (defer→2), +0.004 (defer→3), +0.009 (defer→4)** — the rank gain (~+0.030 at turn 4) is mostly eaten by the 0.02/turn efficiency cost.

So gating's ceiling under the starter ranking is **under one point**. Design (implemented last, only if still justified):

- Returning fewer than 10 recommendations is legal (evaluator scores "the first 10 valid unique" — fewer is fine).
- Turn 1–2 with large/flat-scored pool: return only the top 1–3. Confidence high or pool tiny (≤ ~12): return the full ranked 10. In intent_override sessions, turns 1–2 are hit-ineligible anyway (§1 fact 13), so gating there is free.
- From turn ~5 on, and always by turn 8: return the full 10 regardless — HitRate protection dominates (a miss forfeits ~0.7).
- **Decision procedure:** re-run the shadow evaluator after Stage 1 (the reranker changes the curves in both directions — better late ranks, but also better turn-1 ranks, which shrinks the gap). Ship only if the re-measured ceiling is still positive AND it wins on the validation split with zero hit-rate loss.

### 5.5 Override recovery and buyer dissatisfaction

**Override recovery** (the 5.3 → ~4 MTTC lever):
- On the override message: extract `new_value` verbatim, exact-match it immediately, set `p_buy` = 0.9.
- **Keep the pre-override constraints as `demoted_evidence`** (§1 fact 10) behind a consistency gate: if candidates matching *new + old* exist, boost them (on this evaluator that intersection is often exactly the target); if the intersection is empty, the old evidence is genuinely contradicted — ignore it. Safe in both worlds: if the private set makes overrides truly contradictory, the gate self-disables.
- Keep asking `other` after the override — the card usually still holds undisclosed constraints (including, amusingly, the "ignored" one, which the simulator will happily re-reveal).

**"Those options are not quite right yet"** fires when we return `ask_attribute: null` without a hit — under §5.2 rule 3 that happens only once the card is drained, and it is the *designed* strategy-switch trigger:
- Strategy switch: **now, and only now,** penalize `stale_recs`; relax the *softest* constraint (feature < style < use_case < color < material < budget ≈ category); resume asking `other`.
- **The stale-rec penalty lives here and nowhere else — never in the default rerank path.** Reason (§1 fact 13): the evaluator's hit check gates on `override_applied`, so in intent_override sessions the target itself can be among the shown-but-not-hit recommendations of turns 1–2. An unconditional shown-item penalty would compound every turn and push the target down in exactly the sessions we most need to recover — a silent MRR loss no aggregate metric would attribute. `stale_recs` is cleared on the override message (§4) and the penalty applies only after this dissatisfaction reply; even then it is safe, because post-drain a shown-but-not-hit list provably excludes the target in non-override sessions and the override has already fired in override ones. Micro-test in §8.
- If exact-phrase retrieval returns 0 products (over-constrained or paraphrased), fall back per-constraint: each constraint alone → BM25 on its terms → dense.

## 6. Build stages

**Stage 0 — Harness sanity (DONE, 2026-08-29).** Tests green; baseline measured (`results_starter.json`).

**Stage 1 — Exact-constraint rerank (biggest win, no new deps).**
Capture the full `"what matters is:"` payload; split only on `"; "`; **strip the sentence-final period from the last segment** (§1 fact 11); store verbatim; build the shared normal form for constraints *and* product text (§1 fact 11: colon-flattening for details, punctuation→space, keep `$ % .` in numbers); strip the `"color: "` generated prefix; budget via permissive numeric regex anywhere in the constraint — no prefix assumption, no number → no price filter (§1 fact 12) — near-equality with range fallback; specificity-weighted scoring (long feature strings ≫ single words); dominant boost for all-constraints-matched; exact category-tail filter from turn 1 (mechanism confirmed: the starter's loose ±soft-weight terms are what drifts — §1 fact 4).
*Accept:* public MRR ≥ 0.85, no scenario hit-rate regression, micro-tests of §8 item 3 green. *Effort:* ~1 day.
> **DONE (2026-08-29, `results_stage1.json`):** score 0.896, **hit-rate 1.000 on all four scenarios**, MTTC 1.55, override MTTC already 3.6. MRR landed at 0.690, not 0.85 — a verified structural ceiling, not a weight problem: the anchor guarantees the target in the pool from turn 1, so 140/200 sessions hit before constraints exist to rank on (weight scaling reproduced identical metrics to six decimals). Resolved by §5.4 gating in Stage 3.

**Stage 1.5 — Paraphrase stress harness (moved up from Stage 4 — measure before building the fix).**
Wrap `initial_message`/`customer_reply` output with a paraphraser and rerun the evaluator — our own private-set proxy. Built now so Stages 2–6 all report clean **and** paraphrased numbers; every ablation row carries both for free. The LLM paraphraser writes its rewrites to a cached corpus (`data/paraphrase_cache.jsonl`) keyed by original text, so the harness replays offline — otherwise the harness itself would violate §9's no-network final verification. Include budget phrasing variants ("around fifty dollars", "my budget's about $80") and reordered constraint lists.
*Accept:* harness replays from cache with no network; Stage 1 clean + paraphrased scores recorded. *Effort:* ~0.5 day.
> **DONE (2026-08-29, `tools/paraphraser.py` + `tools/paraphrase_eval.py`):** starter collapses 0.859 → 0.664 (L1) / 0.616 (L2), and the damage is dominated by **frame-parsing brittleness** (the "looking for"/override/reply regexes), not loss of the constraint substring signal. This redirected Stage 2 (permissive extraction became a requirement) and pre-decided Stage 4's scope.

**Stage 2 — Question policy + override recovery.**
Always-`other` drain (§5.2, with the A/B against the old ladder as the ablation row); exhausted tracking; never brand/category; never `null` before `drained ≥ 2`; override → demoted evidence + consistency gate + immediate exact-match of new value + `stale_recs` cleared.
*Accept:* override MTTC ≤ 4.2, overall MTTC ≤ 2.9, A/B documented. *Effort:* ~1 day.
> **DONE (2026-08-29, `results_stage2.json`):** clean run byte-identical to Stage 1 (the generalized patterns parse the exact frames identically — robustness came free); under paraphrase the agent loses **0.24–0.38 points** where the starter lost 19.5–24.3, hit-rate 1.0 held at both levels. Override MTTC 3.6 ≤ 4.2 ✓, MTTC 1.55 ≤ 2.9 ✓. Key mechanisms: token-suffix anchor scan, payload family regex + colon-joiner heuristic, demotion asymmetry as the consistency gate, memo invalidation on override.

**Stage 3 — Soft routing, priors, diversification (independently gated features).**
The old bundle ("ship six features, accept if TechnicalScore ≥ 0.93") could not attribute a regression to a feature. Each feature now lands as its own commit with its own before/after measurement (clean + paraphrased) and its own gate:

| Feature | Ships only if |
|---|---|
| `p_buy` blend (§5.1) | no regression vs the saturated binary route on public; paraphrased-run delta reported (robustness is its whole purpose) |
| profile-tag prior + explanations | MRR delta ≥ 0 — plausibly pure noise; cut without ceremony if flat (explanation strings can stay in `message` regardless) |
| MMR diversification (§5.3) | hit-rate delta ≥ 0 **and** MRR delta ≥ −0.005. Pure downside risk for MRR (any reorder can only move a ranked target down): strength scales with `1 − p_buy`, disabled outright when `p_buy > 0.8` |
| relaxation ladder (§5.5) | fires only on dissatisfaction; tested in isolation with a forced-dissatisfaction session |
| stale-rec penalty (§5.5) | conditional path only (§1 fact 13); the §8 no-penalty-without-dissatisfaction micro-test green |
| rec gating (§5.4) | **last**, and only if the post-Stage-1 shadow re-run still shows a positive ceiling; adopt on validation split with zero hit-rate loss |

*Accept:* public TechnicalScore ≥ 0.93 with the per-feature table filled in; every cut feature documented with its number. *Effort:* 1–2 days.
> **DONE (2026-08-29, `results_stage3.json`):** score **0.962** clean / 0.962 L1 / 0.959 L2, hit-rate 1.0 everywhere, MRR 0.947. Shipped: rec gating (**+0.0662** — see §5.4 outcome note; tune 151 / validate 49, guardrail passed), stale-rec penalty and relaxation ladder as dissatisfaction-only insurance, demo guard. Cut with measured numbers: p_buy gating escape (−0.0027 — releasing full lists early re-creates lock-ins), profile-tag prior (MRR +0.0025 but net −0.00135 on score via MTTC — cut on the honest net gate, not the literal MRR gate), MMR (exactly 0 — its firing condition barely occurs; pure private-set downside). Cut code stays behind off-by-default `FEATURE_*` flags with deltas recorded.

**Stage 4 — Dense retrieval (the paraphrase fix, sized by Stage 1.5's measurements).**
Precompute embeddings offline (`sentence-transformers/all-MiniLM-L6-v2` or similar) → ship `embeddings.npy` + id list; runtime = numpy matmul only. Query = revealed constraints + slot summary. Scope it to whatever degradation Stage 1.5 actually measured — if the lexical path holds up under paraphrase, this shrinks to a thin insurance layer. Size check: submission rules set no numeric cap but say "lightweight local assets" — 38 MB float16 is defensible; fallback to int8 quantization (~19 MB) or a smaller model if organizer guidance tightens.
*Accept:* TechnicalScore drop ≤ 3 points under the Stage 1.5 harness. *Effort:* ~1 day.
> **SKIPPED on the measurements it was scoped by (2026-08-29):** after Stage 2's robust extraction, the L2 residual is **~0.3 points with hit-rate 1.0** — the accept criterion is already met with no dense layer at all. Honest caveat (also in `REPORT.md`): the harness shares authorship with the agent; an adversarial paraphraser that rewrote *constraint payloads semantically* (synonyms, unit changes) would evade lexical matching, and that residual risk remains unmeasured. Dense retrieval stays the documented contingency if organizer guidance signals semantic rewording.

**Stage 5 — LLM extraction layer (optional, keyed).**
Small fast model (e.g. Haiku) for slot/constraint-span extraction + intent classification + message phrasing. Strict JSON out, regex fallback on any error/timeout, honest `usage`, `USE_LLM` flag documented.
> **SKIPPED (2026-08-29):** the deterministic extractor holds hit-rate 1.0 under both paraphrase levels, leaving the LLM layer nothing to earn. The 0-token headline is stronger than a marginal robustness layer; remains the natural extension if a private set defeats the regex families.

**Stage 6 — Tuning with an overfit guard.**
Split the public 200 into 150 tune / 50 validate (stratified by scenario). Grid/greedy-tune the few thresholds that genuinely need it on tune only; never tune on validate. **Honesty constraint on claims:** a stratified 50-session validate split holds ~7–8 intent_override and ~2–3 boundary sessions — enough for an aggregate no-regression guardrail, useless for per-scenario conclusions. So: validate is used *only* as an aggregate-score guardrail; per-scenario numbers (including the override MTTC 5.3 → ~4 claim) are reported on the full 200 with the overfit risk stated plainly in the report. Never present a validate-split per-scenario delta as evidence of generalization.
> **DONE (2026-08-29, folded into Stage 3):** only the gating thresholds were tuned (tune 151 / validate 49, stratified every-4th); validate guardrail passed (0.9642, hit-rate 1.0, no loss); shipped config's tune-set edge over its nearest rival was one session/one turn (0.00013) — chosen for deferral headroom, noise either way.

**Stage 7 — Report + demo.**
Ablation table (score after each stage × per scenario), §2.1 concept-mapping story, cost/latency/token table (deterministic path = 0 tokens), limitations, one demo session — pick an **intent_override** session: it shows extraction, override demotion, consistency gating, re-asking, and rank-1 conversion in one transcript.
> **DONE (2026-08-29):** `REPORT.md` at repo root (method, stage + per-feature ablations, both measurement instruments, limitations, cost/latency, reproduction) with the intent_override demo transcript from `tools/demo_session.py`; `requirements.txt` documents the stdlib-only runtime.

## 7. Standing out: the judging narrative

The spec's "Innovation Directions" list is effectively the rubric for non-score judging. Map every line to a shipped feature with evidence:

| Spec innovation direction | Our feature | Evidence in report |
|---|---|---|
| Buying vs Browsing routing, multi-route retrieval | Soft intent routing (`p_buy`) + cascade | blend-vs-binary comparison, per-scenario table |
| Hybrid retrieval + semantic reranking | exact → FTS5 BM25 → dense cascade | Stage 1.5 clean-vs-paraphrased curves + Stage 4 ablation |
| Structured constraint state, override handling, dynamic context | slot store, verbatim constraints, demoted evidence + consistency gate | override MTTC 5.3 → ~4 before/after |
| Adaptive clarification, question-value estimation | always-`other` drain, *proven* weakly dominant from the simulator code, + A/B vs the entropy ladder | dominance argument + MTTC ablation row |
| Safe personalization from aggregate profile | preference-tag prior + tag-aware explanations | example transcripts |
| Failure detection, strategy switching | drained detection, relaxation ladder, stale-rec penalty | boundary/miss case studies |
| Low latency, low token cost | deterministic core: **0 tokens, ~0.86+ alone**; LLM optional | cost/latency table |
| Transparent recommendation explanations | "because you need X and care about Y" in `message` | demo session |

Differentiators no scoreboard shows but judges will notice: the **measured ablation table** (each feature's contribution, clean + paraphrased), the **paraphrase stress harness** (we built our own private-set proxy instead of hoping), the **shadow evaluator** (`tools/shadow_evaluator.py` — we measured the rank-vs-turn counterfactual instead of arguing from a confounded comparison, and demoted our own gating idea when the numbers came back small), the **tune/validate split** (we can claim the aggregate score generalizes), and the honest **"attention → three scalars"** design story (§2.1) — evidence of choosing the right abstraction, which is rarer than adding machinery.

## 8. Verification

1. `python3 -m unittest discover -s tests` green at every stage.
2. `python3 -m evaluator.local_evaluator --output results.json` after each stage (defaults point at `data/`); log the four scenario breakdowns, not just the aggregate. Keep one `results_stageN.json` per stage — clean and paraphrased — for the ablation table. Re-run `python3 -m tools.shadow_evaluator` after any reranker change: the rank-vs-turn curves drive the §5.4 gating decision.
3. Targeted micro-tests: override demotes-not-deletes and the consistency gate flips both ways; no-preference never re-asked; verbatim constraint survives round-trip; normal-form matching catches the `"key: value"`/`"key value"` case; **a two-constraint reply matches both constraints** (trailing-period strip — §1 fact 11); **a deliberately mid-word-truncated constraint still matches** (§1 fact 11); **budget extracts from paraphrased phrasings and yields no filter when no number is present** (§1 fact 12); budget point-match; `null` never returned before `drained ≥ 2`; **no stale-rec penalty is ever applied without a preceding dissatisfaction reply** — specifically, a session where the target was shown-but-not-hit (pre-override) must score it no lower on the following turn (§1 fact 13); schema-valid responses on turns 1–10 with no exceptions; rec-gating always full-10 by turn 8.
4. Overfit guards: the exact-substring trick is justified *by construction* (intent cards are derived from product metadata — spec), but every exact-match feature must have a lexical/dense fallback exercised in tests, and every tuned threshold is validated on the held-out 50 as an aggregate guardrail (§6 Stage 6) and under the paraphrase harness (§6 Stage 1.5).

## 9. Submission constraints to remember

- Offline final scoring is possible → the deterministic fallback path must produce a full run with no network; document `USE_LLM` on/off behavior explicitly.
- Dependencies declared in `requirements.txt`; numpy is safe; sentence-transformers needed only at *build* time (runtime needs numpy alone).
- No API keys in the repo; env vars only. Disclose model, token cost, latency in the report.
- Deliverables: agent + helpers, setup instructions, short report, one demo multi-turn session (use an intent_override session, §6 Stage 7).
