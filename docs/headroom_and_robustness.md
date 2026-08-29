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

## Reproduction

```bash
python3 -m tools.l3_eval --agent submission.agent --level L3a --seeds 0,1,2
python3 -m tools.l3_eval --agent submission.agent --level L3b --seeds 0,1,2
python3 -m tools.l3_eval --agent submission.agent --level L3a --ablate anchor
python3 -m tools.l3_eval --agent submission.agent --level L3a --mutate-singles
python3 -m tools.shadow_evaluator --agent submission.agent   # rank-vs-turn curves
```
