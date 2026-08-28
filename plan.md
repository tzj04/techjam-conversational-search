# Plan: Conversational Search Agent

## TL;DR

Build a **slot-based constraint state machine** with a **cascading retriever** (exact-phrase → BM25 → dense embeddings), an **information-gain question policy**, and **soft intent routing**: a continuous buyer-confidence scalar that blends constraint-driven scoring with profile/popularity priors, instead of a hard buyer/browser branch. An LLM is used only for robust intent/slot extraction and message phrasing (with a deterministic regex fallback so the agent runs fully offline). No attention-mechanism machinery — but the *concepts* behind that idea (weighted evidence, recency dominance, confidence-blended recommendations) survive as three scalar weights in the reranker; see §2.1.

**Measured baseline (current `starter/agent.py`, public set, 2026-08-29):** TechnicalScore **0.859**, HitRate@10 **0.975**, MRR **0.710**, MTTC **3.08**. Per scenario: buying 0.99 hit / MTTC 2.6, browsing 0.975 / 2.7, boundary 1.0 / 2.6, intent_override **0.933 / 5.3**. (Run: `python3 -m evaluator.local_evaluator --catalog catalog.jsonl` — catalog is at repo root, not `data/`. Ignore the weak_bm25 0.107 reference in `docs/baseline_results.json`.)

The three levers, in order of points available:

1. **MRR 0.71 → ~0.9** (30% of score). Two causes, both measured in `results_starter.json`:
   - **Rank lock-in:** 71 of 195 hits happen on **turn 1** at mean rank **2.8** — the session ends at first hit, freezing a mediocre rank before any constraint was revealed. Hits from turn ≥3 average rank **2.0**. Fix: exact-constraint reranking (§6 Stage 1) + early-turn recommendation gating (§5.4).
   - **Fuzzy matching:** the starter tokenizes revealed constraint strings, destroying the exact-substring signal (§1 fact 1).
2. **Intent-override MTTC 5.3 → ~4** (floor ≈ 3.5, override fires turn 3–4). We currently lose ~1.8 turns re-converging. Fix: §5.5 — including the discovery that the "old" preference is still true of the target (§1 fact 10).
3. **Protect HitRate 0.975 on the private set**, which may paraphrase customer messages — the BM25 → dense fallback cascade (§6 Stage 4) plus our own paraphrase stress harness (§8).

---

## 1. What the evaluator actually does (read this first — it drives every design choice)

From `evaluator/local_evaluator.py` and `docs/competition_specification.md`:

1. **The hidden target's "intent card" is built from its own metadata.** `hard_constraints` / `soft_preferences` are literal (≤180-char) strings from the product's `features`, `details`, detected material, color, and price. When the customer says *"For that, what matters is: 100% cotton; Machine wash"*, those are **exact substrings of the target product's catalog text**. Exact-phrase search on a specific revealed constraint often narrows 50k products to a handful — our strongest signal by far.
2. **Constraints are revealed only in response to `ask_attribute`.** `customer_reply()` returns up to 2 undisclosed constraints whose `classify_constraint()` matches the asked attribute. Asking `other` matches **any** constraint type → highest information gain per turn. Asking `brand` or `category` reveals nothing (the classifier never returns those types).
3. **Returning `ask_attribute: null` before a hit wastes a turn** — the customer replies "ask me about one specific attribute." So: always ask something until the card is drained.
4. **The first message gives the coarse category** — literally the last 2 non-generic entries of the target's `categories` field, joined ("I'm looking for {Women Dresses}..."). Exact category-tail match is a strong hard-ish filter, better than treating them as loose terms.
5. **Scenario is identifiable from message shape** (see §5.1).
6. **Only the rank of the exact `parent_asin` matters**, and per-session score algebra says rank dominates speed: a hit is worth ~0.5, rank 1 vs rank 3 is worth 0.2, one extra turn costs only 0.02. **A rank-1 conversion is worth ~10 turns of delay.** Optimize rank first, speed second, but never risk the hit itself (a miss forfeits ~0.7).
7. **Exceptions/timeouts = miss.** Everything gets try/except and a deterministic fallback.
8. **The private evaluator may paraphrase customer messages** ("paraphrasing cannot decide correctness" — but it can change surface forms). Every exact-match feature needs a lexical → dense fallback path.
9. **The session ends at the FIRST hit** (`local_evaluator.py:252-255`). Combined with fact 6, this is the rank lock-in problem from the TL;DR: an early top-10 appearance at rank 5 permanently costs 0.24 vs converging to rank 1 two turns later. This motivates deliberately *limiting* early recommendations while uncertainty is high (§5.4).
10. **The intent-override "old" preference is still true of the target.** `behavior_for()` sets `old_value = soft_preferences[-1]` and `new_value = hard_constraints[0]` — *both from the same target's card*. The override message ("ignore my earlier preference") replaces a soft preference with a hard constraint of the *same product*. The old string still exact-matches the target — and it is never marked disclosed, so a later `ask` can even re-reveal it. **Don't discard it: demote it to low-weight evidence** with a consistency gate (§5.5). This is the key to closing the override MTTC gap.
11. **Exact-substring matching needs a shared normal form — naive `in` checks silently fail.** The evaluator flattens `details` dicts as `"key: value"` but the agent's product text joins them as `"key value"` (no colon); internal whitespace isn't collapsed on the product side; constraints are truncated at 180 chars (possibly mid-word); reply constraints are joined with `"; "` and suffixed `"."`. Implementation: normalize both sides (lowercase, collapse all punctuation runs to single spaces, keep `$ % .` inside numbers), split replies **only** on `"; "`, strip generated prefixes (`"color: "`, `"budget around $"`), and use plain substring (not word-boundary) matching so truncated tails still match.
12. **Two more literal signals hiding in the card generator:** `"budget around $X"` means X **is the target's exact price** — treat as near-equality (±cents), with a ±35% range fallback for private-set safety. And a bare material word as the first constraint means the material regex matched the target's corpus — single-word constraints match thousands of products, so **weight each constraint by its specificity** (length / token count), not uniformly.

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
        │                 − exclusions − stale-rec penalty
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
    stale_recs: set[str]               # asins already shown (small penalty)
    drained: int                       # consecutive "no additional preference" replies
    budget_min/max/point: float | None # point ≈ exact target price when card-style
```

Rules:

- **Override handling** (black → blue and the scripted override): a new value for a filled slot *replaces* it. The explicit override message ("actually / ignore my earlier preference / instead") moves current soft slots and `revealed_constraints` into `demoted_evidence` (weight ~0.1× — see the consistency gate in §5.5), keeps `category_terms`, extracts the new requirement verbatim, and sets `p_buy` high.
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

Question value = how much the answer will shrink/re-rank the pool, given what the simulator can actually reveal:

1. **`other` first** (turns 1–2): reveals up to 2 constraints of *any* type — maximal drain of the intent card. Two `other` asks usually empty a 4-constraint card by turn 3.
2. Then the unfilled, non-exhausted attribute with **highest value-diversity over the current top-~80 candidates** (entropy ≈ expected pool reduction). The starter's `_attribute_diversity()` is a good base.
3. **Never ask:** attributes in `exhausted`, filled slots, or `brand`/`category` (the classifier never produces those — guaranteed wasted turn).
4. **Stop asking** (`ask_attribute: null`) only when `drained ≥ 2` — the card is empty; from then on rely on ranking + diversification. Never return `null` before that (§1 fact 3).

### 5.3 Top-10 construction

Rank by target-probability, then **diversify across residual uncertainty** instead of returning 10 near-duplicates:

- Constraint-satisfying candidates first, sorted by blended score.
- For attributes we *don't* know (unfilled or "no preference"), spread slots across values (unknown color → don't return 10 black shoes). Cheap MMR-style pass: penalize a candidate sharing all unknown-attribute values with an already-picked one. Diversification strength scales with `1 − p_buy` and pool entropy.
- Dedupe `parent_asin`, catalog-valid ids only, at most `top_k`.

### 5.4 Early-turn recommendation gating (the MRR lock-in fix)

Evidence: 71/195 baseline hits land on turn 1 at mean rank 2.8, ending those sessions at RR ≈ 0.36 when two more turns of constraints would have produced rank ~1 (turn-≥3 hits average rank 2.0). Since a rank-1 conversion is worth ~10 turns of delay (§1 fact 6), **deliberately show fewer recommendations while the pool is still ambiguous**:

- Returning fewer than 10 recommendations is legal (evaluator scores "the first 10 valid unique" — fewer is fine).
- Turn 1–2 with large/flat-scored pool: return only the top 1–3 (a hit there is a *good* hit — rank ≤3). Confidence high or pool tiny (≤ ~12): return the full ranked 10.
- From turn ~5 on, and always by turn 8: return the full 10 regardless — HitRate protection dominates (a miss forfeits ~0.7).
- Tune the thresholds on the tune split only (§8) and adopt **only if** it wins on the validation split with no hit-rate loss. This is an experiment with a guardrail, not a leap of faith.

### 5.5 Override recovery and buyer dissatisfaction

**Override recovery** (the 5.3 → ~4 MTTC lever):
- On the override message: extract `new_value` verbatim, exact-match it immediately, set `p_buy` = 0.9.
- **Keep the pre-override constraints as `demoted_evidence`** (§1 fact 10) behind a consistency gate: if candidates matching *new + old* exist, boost them (on this evaluator that intersection is often exactly the target); if the intersection is empty, the old evidence is genuinely contradicted — ignore it. Safe in both worlds: if the private set makes overrides truly contradictory, the gate self-disables.
- Keep asking `other` after the override — the card usually still holds undisclosed constraints (including, amusingly, the "ignored" one, which the simulator will happily re-reveal).

**"Those options are not quite right yet"** only happens when we returned `ask_attribute: null` without a hit — with §5.2 rule 4 it should never fire; treat it as a failure-detection trigger if it does:
- Strategy switch: penalize `stale_recs`, relax the *softest* constraint (feature < style < use_case < color < material < budget ≈ category), re-ask the highest-entropy attribute.
- If exact-phrase retrieval returns 0 products (over-constrained or paraphrased), fall back per-constraint: each constraint alone → BM25 on its terms → dense.

## 6. Build stages

**Stage 0 — Harness sanity (DONE, 2026-08-29).** Tests green; baseline measured (`results_starter.json`).

**Stage 1 — Exact-constraint rerank (biggest win, no new deps).**
Capture the full `"what matters is:"` payload; split only on `"; "`; store verbatim; build the shared normal form for constraints *and* product text (§1 fact 11: colon-flattening for details, punctuation→space, keep `$ % .` in numbers); strip `"color: "` / `"budget around $"` generated prefixes (budget → near-equality price filter, range fallback); specificity-weighted scoring (long feature strings ≫ single words); dominant boost for all-constraints-matched; exact category-tail filter from turn 1.
*Accept:* public MRR ≥ 0.85, no scenario hit-rate regression. *Effort:* ~1 day.

**Stage 2 — Question policy + override recovery.**
`other`-first ordering; exhausted tracking; never brand/category; never `null` before `drained ≥ 2`; override → demoted evidence + consistency gate + immediate exact-match of new value.
*Accept:* override MTTC ≤ 4.2, overall MTTC ≤ 2.9. *Effort:* ~1 day.

**Stage 3 — Soft routing, priors, diversification, gating.**
`p_buy` blend (§5.1); profile-tag prior + explanation strings in `message`; MMR diversification; stale-rec penalty; relaxation ladder; **rec-gating experiment** (§5.4) adopted only on validation-split evidence.
*Accept:* public TechnicalScore ≥ 0.93; gating decision documented either way. *Effort:* 1–2 days.

**Stage 4 — Dense retrieval + paraphrase stress harness (robustness).**
Precompute embeddings offline (`sentence-transformers/all-MiniLM-L6-v2` or similar) → ship `embeddings.npy` + id list; runtime = numpy matmul only. Query = revealed constraints + slot summary. Then build the **paraphrase harness**: wrap `customer_reply`/`initial_message` output with an LLM paraphraser and rerun the evaluator — our own private-set simulator.
*Accept:* TechnicalScore drop ≤ 3 points under paraphrase. *Effort:* ~1 day.

**Stage 5 — LLM extraction layer (optional, keyed).**
Small fast model (e.g. Haiku) for slot/constraint-span extraction + intent classification + message phrasing. Strict JSON out, regex fallback on any error/timeout, honest `usage`, `USE_LLM` flag documented.

**Stage 6 — Tuning with an overfit guard.**
Split the public 200 into 150 tune / 50 validate (stratified by scenario). Grid/greedy-tune rerank weights and gating thresholds on tune only; report both splits. Never tune on validate.

**Stage 7 — Report + demo.**
Ablation table (score after each stage × per scenario), §2.1 concept-mapping story, cost/latency/token table (deterministic path = 0 tokens), limitations, one demo session — pick an **intent_override** session: it shows extraction, override demotion, consistency gating, re-asking, and rank-1 conversion in one transcript.

## 7. Standing out: the judging narrative

The spec's "Innovation Directions" list is effectively the rubric for non-score judging. Map every line to a shipped feature with evidence:

| Spec innovation direction | Our feature | Evidence in report |
|---|---|---|
| Buying vs Browsing routing, multi-route retrieval | Soft intent routing (`p_buy`) + cascade | blend-vs-binary comparison, per-scenario table |
| Hybrid retrieval + semantic reranking | exact → FTS5 BM25 → dense cascade | Stage 4 ablation + paraphrase stress test |
| Structured constraint state, override handling, dynamic context | slot store, verbatim constraints, demoted evidence + consistency gate | override MTTC 5.3 → ~4 before/after |
| Adaptive clarification, question-value estimation | `other`-first + entropy-based attribute choice | MTTC ablation |
| Safe personalization from aggregate profile | preference-tag prior + tag-aware explanations | example transcripts |
| Failure detection, strategy switching | drained detection, relaxation ladder, stale-rec penalty | boundary/miss case studies |
| Low latency, low token cost | deterministic core: **0 tokens, ~0.86+ alone**; LLM optional | cost/latency table |
| Transparent recommendation explanations | "because you need X and care about Y" in `message` | demo session |

Differentiators no scoreboard shows but judges will notice: the **measured ablation table** (each stage's contribution, per scenario), the **paraphrase stress harness** (we built our own private-set proxy instead of hoping), the **tune/validate split** (we can claim the score generalizes), and the honest **"attention → three scalars"** design story (§2.1) — evidence of choosing the right abstraction, which is rarer than adding machinery.

## 8. Verification

1. `python3 -m unittest discover -s tests` green at every stage.
2. `python3 -m evaluator.local_evaluator --catalog catalog.jsonl --dataset data/public_set.jsonl --output results.json` after each stage; log the four scenario breakdowns and the tune/validate split, not just the aggregate. Keep one `results_stageN.json` per stage for the ablation table.
3. Targeted micro-tests: override demotes-not-deletes and the consistency gate flips both ways; no-preference never re-asked; verbatim constraint survives round-trip; normal-form matching catches the `"key: value"`/`"key value"` case; budget point-match; `null` never returned before `drained ≥ 2`; schema-valid responses on turns 1–10 with no exceptions; rec-gating always full-10 by turn 8.
4. Overfit guards: the exact-substring trick is justified *by construction* (intent cards are derived from product metadata — spec), but every exact-match feature must have a lexical/dense fallback exercised in tests, and every tuned threshold is validated on the held-out 50 (§6 Stage 6) and the paraphrase harness (§6 Stage 4).

## 9. Submission constraints to remember

- Offline final scoring is possible → the deterministic fallback path must produce a full run with no network; document `USE_LLM` on/off behavior explicitly.
- Dependencies declared in `requirements.txt`; numpy is safe; sentence-transformers needed only at *build* time (runtime needs numpy alone).
- No API keys in the repo; env vars only. Disclose model, token cost, latency in the report.
- Deliverables: agent + helpers, setup instructions, short report, one demo multi-turn session (use an intent_override session, §6 Stage 7).
