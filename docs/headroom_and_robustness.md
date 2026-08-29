# Where the remaining score is, and where it isn't

Working notes behind the Stage 4 changes. Everything here is measured on the
public 200 with the shipped evaluator; commands are at the bottom.

## 1. The clean set is nearly saturated: 0.0163 of headroom

`results_stage3.json`, decomposed:

| component | now | practical ceiling | score headroom |
|---|---|---|---|
| HitRate@10 | 1.000 | 1.000 | 0 |
| MRR | 0.946929 | 1.000 | **0.0159** |
| MTTC | 2.095 | ~2.08 | ~0.0004 |

**Hit-rate is exhausted.** **MTTC is at its policy floor**, not near it:

| scenario | n | MTTC | floor | why the floor is where it is |
|---|---|---|---|---|
| buying | 80 | 1.625 | ~1.6 | turn-1 hits need rank 1 under depth-1 gating |
| browsing | 80 | 1.887 | ~1.9 | turn 1 carries no constraint at all |
| boundary | 10 | 2.600 | 2.6 | the boundary one-off consumes turn 2 by construction |
| intent_override | 30 | 3.733 | 3.667 | the evaluator blocks conversion until the override fires on turn 3 or 4 |

So ~98% of what is left is MRR, and MRR is 17 sessions that convert below
rank 1. A practical ceiling is **≈0.9785**.

## 2. 15 of those 17 are a hard plateau, not a timing problem

Replaying them through the non-stopping shadow evaluator (rank at every turn,
no early stop):

```
public_0020 buying     conv rank 7 @t3   ranks=[-,7,7,7,7,7,7,7,7,7]
public_0083 buying     conv rank 3 @t3   ranks=[-,3,3,3,3,3,3,3,3,3]
public_0099 browsing   conv rank 4 @t3   ranks=[-,4,4,4,4,4,4,4,4,4]
public_0144 override   conv rank 7 @t5   ranks=[-,10,7,7,7,7,7,7,7,7]
...
public_0050 boundary   conv rank 2 @t3   ranks=[2,2,2,1,1,1,1,1,1,1]   <- improves
public_0187 boundary   conv rank 10 @t3  ranks=[-,-,10,1,1,1,1,1,1,1]  <- improves
```

15 of 17 are **flat across all ten turns**. Every constraint on the card has
been revealed and the rank does not move. No policy change — deferring longer,
asking differently, more turns — reaches them. Only `public_0050` and
`public_0187` would improve by waiting, and tuning the gate for two sessions is
pure overfit.

**Why they plateau.** The target and a set of rivals all match *every*
constraint, so the tiebreak decides — and the tiebreak is the popularity prior:

```
public_0083  constraints = [polyester, 100% Polyester, Imported, Button closure]
             31 of 681 products in the anchor set match all four
             target B0BPMCJ1RD is 3rd among them by rating_number
public_0099  constraints = [cotton, 60% Cotton 40% Polyester, Imported, Drawstring closure]
             4 match all four; target is 4th
```

These constraints are generic Amazon boilerplate. There is no lexical signal
left to separate the target from its rivals — the remaining 0.0159 is not
obviously reachable, and chasing it on 200 sessions is exactly the overfit the
private set would punish.

## 3. The popularity prior is a real structural signal, not a hack

Sessions are sampled from the 5-core leave-last-out split of real purchases, so
targets are systematically popular. Target's percentile **within its own anchor
set**:

| by | mean | median | top decile |
|---|---|---|---|
| `rating_number` | 0.937 | 0.978 | 168 / 197 |
| `average_rating` | 0.617 | 0.639 | — |

This is what carries the agent when constraint matching fails. Ordering the
anchor set by popularity alone puts the target:

| ordering | top-1 | top-10 | MRR |
|---|---|---|---|
| shipped `0.08·rating + 0.05·log1p(n)` | 0.340 | 0.805 | 0.4973 |
| `rating_number` alone | 0.350 | **0.815** | 0.5048 |
| `average_rating` alone | 0.015 | 0.165 | 0.0667 |

**80.5% top-10 is the floor under total constraint failure**, and it is why the
agent degrades rather than collapses. Note `average_rating` contributes nothing
on its own and slightly dilutes the composite.

## 4. L2 is a no-op against this agent — three ways, not two

`tools/paraphraser.py:169` `_mutate_payload`:

- `cat` and `attr` are returned unchanged.
- **`old` is returned unchanged too** — the `intent_override` opening's
  `old_value` is a real constraint (`soft[-1]`) and L2 never touches it.
- `_mutate_constraint` (`:182`) does exactly two things: rephrase a budget
  constraint (bypassed — budget is regex-extracted, never substring-matched)
  or flip the first letter (erased by the agent's lowercasing normal form).

So the L2 row measures frame parsing and payload splitting. It does not test
lexical matching of constraint payloads at all.

## 5. L3: the missing measurement

`tools/l3_paraphraser.py` rewrites payloads *semantically* — meaning-preserving,
so a human still identifies the same product. 60% of constraints mutate.
Single-word constraints (276 of 800, 34.5%: `cotton`, `polyester`, `Imported`)
are left alone by default: any faithful rewording keeps the head noun, and they
are the floor under the degradation. `--mutate-singles` measures the harsher
variant.

| level | score (3 seeds) | hit | MRR | MTTC |
|---|---|---|---|---|
| clean | 0.962179 | 1.000 | 0.947 | 2.095 |
| L2 | 0.959033 | 1.000 | 0.942 | 2.175 |
| **L3a** (payloads rewritten, category tail intact) | **0.887 ± 0.015** | 0.963 | 0.808 | — |
| **L3b** (category tail rewritten too) | **0.832 ± 0.008** | 0.925 | 0.733 | — |

The mechanism: `constraint.norm in text` is all-or-nothing, so a payload
reworded *anywhere* scores exactly as badly as an unrelated product. The
constraint then contributes `−UNMATCHED_PENALTY` to every candidate including
the target, `ALL_MATCHED_BONUS` never fires, and ranking inside the anchor
collapses to the popularity order of §3.

**Caveat, carried deliberately:** this rewriter shares authorship with the
agent, which is the same criticism §6.1 makes of L1/L2. Directional evidence,
not a guarantee.

## 6. A live bug this surfaced: budget misrouting

`_add_constraint` classified a segment as a budget constraint if it contained
any money *word* (`price`, `cost`, `spend`, `dollars`, …). Catalog features and
details are full of them:

```
WELL PRICED, TIMELESS STYLE - Traditional in its design, this inexpensive...
Top present under 30 dollars for a Birthday, Valentine's Day, Mothers day...
```

**479 card-candidate segments across the catalog** matched. When such a string
is revealed as a constraint, the old path threw its text away entirely and
applied a price filter built from whatever digit appeared first — losing the
single most discriminative constraint in that session *and* adding a wrong
prior. One public session (`public_0158`) already hits this; it converts anyway,
so the public cost is zero and the exposure is entirely private-set (~7 of 800
sessions at the same rate).

The guard requires a *price statement* — short (≤64 chars) and number-adjacent —
rather than a money word, and never drops text it declines to route. All five
budget phrasings the simulator and paraphrasers can emit still route correctly;
false routes drop 479 → 46.

Related: **zero budget constraints are revealed on the public set** (0 of 800).
`intent_card` appends `budget around $X` last, after the material and colour
inserts and the features/details entries, so it lands past `cleaned[:4]` for
every one of the 200 targets. Every measurement in the report leaves the entire
budget path untested; the same generator makes it equally unreachable on the
private set.

## 7. The deployment audit: six ways to score zero with every number here unchanged

Everything above measures the *ranking*. This section is the other axis: the
ways the agent stops being an agent at all. Each of these was live, none is
reachable from `tools/run_eval.py`, and each turns a 0.96 into a 0.00 without
any instrument in this repository noticing.

The common shape is that `respond` wraps `_respond` in a bare
`except Exception` returning `recommendations: []`. That is a fail-*silent*
path, not a fail-safe one: any fault anywhere becomes an empty list, and an
empty list is a guaranteed miss.

| # | Trigger | Old behaviour | Now |
|---|---|---|---|
| 1 | `Agent()` constructed from a working directory that is not the repo root | `data/catalog.jsonl` is a **relative** default; `_build_index` returns early, `_products` is empty, every session returns `[]` — no error anywhere | resolve against `$TECHJAM_CATALOG` and the path relative to `submission/agent.py`, which does not move; warn on stderr if still unresolved |
| 2 | One malformed line, or one row without `parent_asin`, in the private catalog | `json.loads` / `KeyError` raises **out of `__init__`** | row skipped, counted in `Agent.skipped_rows` |
| 3 | `average_rating: "4.5 out of 5 stars"` or `rating_number: "1,234"` | `float()` raises out of `__init__` (`_fallback_order` sort) | `_number()` reads the leading numeral; unreadable values sort last |
| 4 | Host SQLite built without FTS5 | `CREATE VIRTUAL TABLE` raises out of `__init__` | `fts_enabled = False`; retrieval degrades to the anchor + popularity floor of §3, which is where 80.5% of the top-10 rate lives anyway |
| 5 | Harness evaluates sessions on a worker thread | sqlite3 connections are thread-affine: `ProgrammingError` on every query, swallowed into `[]` for all 800 sessions | `check_same_thread=False` plus an `RLock` around the query |
| 6 | Any internal fault on any turn | `recommendations: []` — a guaranteed miss | fall back to the session's last good ranking, then to the popularity order, gated exactly as the normal path would gate |

Two of these (#1, #5) depend only on how the organizer *invokes* the agent, and
neither is specified by `docs/agent_api_contract.json`. Reproductions for all
six are in `tests/test_robustness.py`.

### Two more that cost rank rather than everything

**A wrong anchor taken out of a payload.** `_try_opening` runs on every turn
while no anchor is held, and its longest candidate head is the whole message.
So its *token suffix* is the tail of the payload, and a payload ending in a
category name installs that category at the full `ANCHOR_BONUS`. That is
precisely the failure FUZZY_ANCHOR was cut for — "a wrong anchor costs far more
than no anchor" — reintroduced through a different door. Worse, on an
`intent_override` session it also *swallows the override*: `_extract` tried the
opening scan before the pivot test, so `_try_opening` returning `True` meant the
prior evidence was never demoted and the new value never read. Fixed two ways:
every anchor-scan head (suffix, infix and fuzzy alike) is clipped at the payload
separator, and the pivot is tested first whenever there is prior evidence to
override.

This can only fire once the opening's category has failed to resolve, so it is
unreachable on the clean set by construction — which is why no measurement here
found it.

**A tokenizer disagreement on accented text.** The normal form keeps `[a-z0-9]`
only, so it deletes the accented letter and splits the word: `café` → `caf`,
`Damenmütze` → `damenm tze`, `Bébé` → `` (dropped entirely). The FTS5 index is
tokenized `unicode61 remove_diacritics 2`, which folds the accent and keeps the
word whole: `cafe`, `damenmutze`, `bebe`. Every query term built from an
accented constraint therefore matches **nothing**, and the fragments left behind
pollute the document-frequency table that weights partial matching. This is the
same class of bug as the `%`/`.` doc-frequency disagreement already fixed in
Stage 4 — one instance of it was simply left standing. `FEATURE_UNICODE_FOLD`
applies NFD and drops combining marks, which is the identity on ASCII.

A third, latent: `_budget_score` read `price` with a bare `float()`, so a price
shipped as `"$14.99"` was treated as *no price* and **penalised** the one product
it was meant to match exactly. Dead weight today (§6: zero budget constraints
are ever revealed), wrong whenever it is not.

## 8. What is measured, and what is not

`data/catalog.jsonl` is a release download, not repository content, so on a
fresh checkout every end-to-end instrument here is unavailable. The figures in
§§1–6 were taken when it was present; the fixes in §7 were not re-measured
against it.

Two things stand in for that:

- **By construction.** Each §7 fix is a no-op under the conditions every figure
  above was measured in — catalog present, one thread, FTS5 available, no
  exception raised, ASCII text, and (for the anchor and pivot clips) an opening
  whose category resolves on turn 1. None of them can move a number taken under
  those conditions.
- **By regression.** `tools/synthetic_eval.py` runs the shipped evaluator loop
  over a generated catalog of the same *shape* (the templated Amazon metadata of
  §5) with generated sessions in the released scenario mix. Across
  clean/L2/L3a/L3b/catdrift × 5 seeds × 120 sessions — **3,000 sessions** — the
  per-session rank and hit turn are byte-identical before and after. Absolute
  scores there are not comparable to the public-set figures: a generated catalog
  has a far smaller vocabulary, so constraint matching alone finds the target.

What remains unmeasured is the *frequency* of the §7 rank failures on the real
50k catalog: how many category keys occur as the token suffix of a features or
details string, and how much accented text there is. Both are one command away
once the catalog is present:

```bash
python3 -m tools.l3_eval --agent submission.agent --level L3b   --seeds 0,1,2
python3 -m tools.l3_eval --agent submission.agent --level catdrift --seeds 0,1,2
```

`catdrift` is new here and isolates what L3b conflates: it rewrites the category
tail and *nothing else*, leaving payloads byte-identical. REPORT §6.2 names
spec-faithful card construction as the load-bearing assumption, and category
replication is the part of it most likely to drift; under L3b the same rewriter
also rewrites payloads, so category words *inside* a payload drift in lockstep
with the opening's and the interaction above cannot appear.

## Reproduction

```bash
python3 -m unittest discover -s tests                          # 106 tests
python3 -m tools.synthetic_eval --seeds 0,1,2                  # no catalog needed
python3 -m tools.synthetic_eval --level catdrift --seeds 0,1,2
python3 -m tools.l3_eval --agent submission.agent --level L3a --seeds 0,1,2
python3 -m tools.l3_eval --agent submission.agent --level L3b --seeds 0,1,2
python3 -m tools.l3_eval --agent submission.agent --level L3a --ablate anchor
python3 -m tools.l3_eval --agent submission.agent --level L3a --mutate-singles
python3 -m tools.l3_eval --agent submission.agent --level catdrift --seeds 0,1,2
python3 -m tools.shadow_evaluator --agent submission.agent   # rank-vs-turn curves
```
