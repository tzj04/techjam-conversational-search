"""Submission agent: exact-constraint reranking over a category anchor
(Stage 1), question policy, override recovery, and paraphrase-robust
extraction (Stage 2), soft intent routing with recommendation gating,
priors, and failure recovery (Stage 3).

Core ideas (see plan.md):
- Revealed constraints are verbatim substrings of the target's catalog text
  (§1 fact 1), so both sides are mapped through one shared normal form and
  matched with plain substring checks (§1 fact 11).
- The opening message carries the target's coarse category verbatim
  (§1 fact 4); an exact match on the precomputed per-product coarse category
  yields a sticky anchor set that is guaranteed to contain the target on the
  public evaluator, applied as a dominant boost rather than a hard filter.
- Budget constraints encode the target's exact price (§1 fact 12); they are
  scored against price only, never substring-matched.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "have", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "please", "some", "that", "the", "this", "to", "want", "with", "would",
    "you", "your",
}
# Keep $, %, and . only when adjacent to a digit; every other punctuation run
# becomes a single space. Applied identically to product text and constraints.
_NORM_RE = re.compile(r"[^a-z0-9$%.\s]|(?<![0-9])[$%.](?![0-9])")
_BUDGET_WORD_RE = re.compile(r"\b(budget|price|priced|pricing|cost|costs|spend|spending|afford|dollar|dollars|usd)\b")
# A money *word* is not a budget disclosure. 414/50,000 catalog products carry
# one inside a features/details string ("WELL PRICED, TIMELESS STYLE - ...",
# "Top present under 30 dollars for a Birthday..."), and such a string can be
# revealed as an ordinary constraint. Under the bare word test it was routed to
# the budget path, its text discarded, and a bogus price filter applied from
# whatever digit appeared first. A real disclosure is short and number-adjacent.
_BUDGET_PHRASE_RE = re.compile(
    r"\$\s*\d"
    r"|\b(?:budget|price[ds]?|pricing|cost[s]?|spend(?:ing)?|afford)\b[^.]{0,24}?\d"
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:dollars?|usd)\b"
)
BUDGET_MAX_CHARS = 64
_NUMBER_RE = re.compile(r"\$?\d[\d,]*\.?\d*")
# Extraction patterns are written as paraphrase families (Stage 1.5 showed the
# starter's collapse came from frame brittleness), not as copies of any one
# evaluator template.
_PAYLOAD_RE = re.compile(
    r"(?:matters|care about|prioriti[sz]e|priorit\w*|important|must[- ]?haves?"
    r"|needs?|requirements?|essentials?|key things?)[^:]*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_SECONDARY_JOINER_RE = re.compile(r" and also |, plus ")
# Override pivots open the message; scanning only the first 20 chars keeps
# payload text (which starts at char ~21+ in every reply frame) out of reach.
_OVERRIDE_TRIGGER_RE = re.compile(
    r"\b(actually|instead|forget|ignore|scratch|never ?mind|disregard"
    r"|change of plans|on second thought|drop that)\b"
)
_OVERRIDE_VALUE_RE = re.compile(
    r"(?:need|want|require|must have|after)[^:]*:\s*(.+)$", re.IGNORECASE | re.DOTALL
)
_NO_ADDITIONAL_RE = re.compile(
    r"\bno additional preference\b|don'?t have (?:an?y? )?additional preference"
    r"|nothing more to add",
    re.IGNORECASE,
)
_NO_PREF_RE = re.compile(
    r"\bno (?:real )?preference\b|don'?t have a preference|use your judgment"
    r"|your judgment is fine|whatever you think",
    re.IGNORECASE,
)
_DISSATISFIED_RE = re.compile(r"not quite (?:right|it|what)", re.IGNORECASE)
_EXPLORING_RE = re.compile(
    r"still exploring|just browsing|having a look|look(?:ing)? around"
    r"|still weighing|haven'?t settled|window shopping",
    re.IGNORECASE,
)
_BOUNDARY_SPLIT_RE = re.compile(r"\. |\? |! | — |, ")
_ATTRIBUTE_WORDS = [(attribute, attribute.replace("_", " ")) for attribute in ALLOWED_ATTRIBUTES]


def normalize_text(text: str) -> str:
    """Shared normal form for product text and constraints (§1 fact 11)."""
    return " ".join(_NORM_RE.sub(" ", (text or "").lower()).split())


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def extract_budget_point(text: str) -> float | None:
    """Permissive numeric extraction anywhere in the string — never by prefix."""
    match = _NUMBER_RE.search((text or "").replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0).lstrip("$"))
    except ValueError:
        return None


def coarse_category(values: list[str]) -> str:
    """Verbatim copy of the evaluator's category-tail construction."""
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def _flatten_field(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


@dataclass
class _Constraint:
    verbatim: str
    norm: str                    # normalized match string; "" for budget constraints
    weight: float
    is_budget: bool
    budget_point: float | None
    demoted: bool = False        # pre-override evidence kept at low weight (§5.5)


@dataclass
class SessionState:
    profile: dict
    anchor: tuple[str, ...] | None = None
    anchor_set: frozenset[str] = frozenset()
    anchor_bonus: float = 0.0
    category_terms: list[str] = field(default_factory=list)
    loose_terms: list[str] = field(default_factory=list)
    constraints: list[_Constraint] = field(default_factory=list)
    seen_keys: set[str] = field(default_factory=set)
    budget_point: float | None = None
    drained: int = 0                              # consecutive "no additional preference"
    exhausted: set[str] = field(default_factory=set)  # boundary one-off attributes (bookkeeping)
    stale_shown: set[str] = field(default_factory=set)  # asins returned so far this session
    penalized: frozenset[str] = frozenset()       # stale snapshot; only on dissatisfaction (§5.5)
    dissatisfied: bool = False
    p_buy: float = 0.5                            # soft intent confidence (§5.1)
    profile_tags: tuple[str, ...] = ()
    memo: dict[str, list] = field(default_factory=dict)  # asin -> [n_processed, score, n_matched]
    cache_version: tuple | None = None
    cache_ranking: list[str] = field(default_factory=list)


# Rerank weights (Stage 1 — tuned on the public set).
ANCHOR_BONUS = 50.0
ALL_MATCHED_BONUS = 4.0
UNMATCHED_PENALTY = 0.35
BUDGET_EXACT_BONUS = 6.0
BUDGET_RANGE_BONUS = 1.2
BUDGET_MISS_PENALTY = 1.5
BUDGET_NO_PRICE_PENALTY = 0.5

# Stage 3 feature flags. Shipped set per the measured ablation (public 200,
# clean): gating +0.0662; p_buy gating escape −0.0027 (cut); profile prior
# −0.00135 net despite +0.0025 MRR (cut); MMR exactly 0 and pure private-set
# downside risk (cut); stale penalty & relaxation fire only on dissatisfaction,
# zero-cost on clean, kept as failure insurance. Code for cut features is
# retained behind the flags for the ablation record.
FEATURE_GATING = True
FEATURE_PBUY = False
FEATURE_PROFILE_PRIOR = False
FEATURE_MMR = False
FEATURE_STALE_PENALTY = True
FEATURE_RELAXATION = True

# Stage 4 (robustness). Each targets a *named* failure mechanism measured by
# tools/l3_eval.py, not a tuned constant. See REPORT §6.
#
# PARTIAL_MATCH: exact-substring matching is all-or-nothing, so a payload
#   reworded anywhere ("Rubber sole" -> "soles made of Rubber") scores exactly
#   as badly as an unrelated product. Falls back to IDF-weighted token coverage.
# FIELD_WEIGHT: the evaluator builds intent cards from `features` + `details`
#   only, so a constraint found in those fields is stronger evidence than the
#   same string buried in a long `description`.
# INFIX_ANCHOR: the anchor scan reads the category as a token *suffix* of a
#   clause, so any frame that puts words after it with no delimiter loses the
#   anchor even though the category is present verbatim ("I'm shopping for
#   {cat} but haven't settled on anything"). That is 54 of 200 openings at L3a.
#   An exact scan over contiguous token windows recovers 54/54 with 0 wrong
#   picks -- it is still an exact key lookup, so unlike a fuzzy match it cannot
#   land on the wrong category.
# FUZZY_ANCHOR: the anchor scan is an exact key lookup; if the category tail is
#   reworded the anchor is lost entirely rather than degraded. Measured as a
#   *scoring* fallback it was actively harmful (L3b 0.832 -> 0.707): on reworded
#   categories the best-overlap key is the wrong set 57 times in 200, and a
#   wrong anchor bonus buries the target under a whole wrong category. A wrong
#   anchor costs far more than no anchor. So the two roles the anchor plays are
#   split: under uncertainty it feeds candidate *generation* only and never the
#   score, which makes a wrong pick cost nothing but recall.
# REGIME_ESCAPE: the gate's own "informed" test counts *exact* matches, so under
#   paraphrase it fails in the same correlated way and the agent sits at depth 1
#   until GATE_CAP_TURN. Detects "constraints exist but nothing matches" and
#   opens the gate.
FEATURE_BUDGET_GUARD = True
FEATURE_PARTIAL_MATCH = True
FEATURE_PARTIAL_IDF_FLOOR = True
FEATURE_FIELD_WEIGHT = True
FEATURE_INFIX_ANCHOR = True
# CUT on measurement. As a scoring fallback it was catastrophic (L3b 0.832112
# -> 0.674195): the best-overlap key is the wrong set 57 times in 200, and a
# wrong anchor bonus buries the target under a whole wrong category. Redesigned
# to feed candidate generation only, it recovered 146 of those 158 points but
# is still net negative where it exists to help -- isolated, L3b 0.832112 ->
# 0.820135; in combination, removing it takes L3b 0.871357 -> 0.875537 with
# clean and L3a unchanged to six decimals. Together with the multi-query result
# (also candidate-adding, also ~-0.012 at L3b) the conclusion is that the pool
# is not usefully recall-limited: extra candidates cost more in precision than
# the recovered targets are worth. INFIX_ANCHOR gets the same 54 openings back
# by exact lookup, with no wrong picks and no pool growth at all.
FEATURE_FUZZY_ANCHOR = False
FEATURE_REGIME_ESCAPE = True
# MULTI_QUERY: retrieval fires a single FTS query built as an OR of up to 40
#   terms, so BM25 favours documents matching many *common* terms and a
#   constraint whose vocabulary is rare can contribute no candidates at all.
#   At L3b that is 5 of 12 remaining misses, where the target never enters the
#   pool. One query per constraint guarantees each contributes independently.
#   CUT on measurement: it does fix recall, but the extra candidates cost more
#   in precision than the recovered targets are worth. Holding everything else
#   constant, L3b fell 0.871357 -> 0.858485 with hit 0.935 -> 0.915, while
#   clean and L3a were unchanged to six decimals. Diagnosing the cause
#   correctly (5 of 12 L3b misses never enter the pool) did not make the
#   obvious remedy net-positive. Retained behind the flag with its delta.
FEATURE_MULTI_QUERY = False
MULTI_QUERY_LIMIT = 120

# Partial matching. Sub-additive in coverage so a full substring match always
# dominates any partial one; below PARTIAL_MIN coverage nothing is awarded.
PARTIAL_SCALE = 0.55
PARTIAL_MIN = 0.34
# Coverage is a *ratio*, so matching one of two common words ("made", "of")
# scores 0.5 while carrying no information. Partial evidence must also be
# discriminative in absolute terms: the matched tokens must carry at least the
# information of one token appearing in `PARTIAL_IDF_QUANTILE` of the catalog.
# Stated as a rule about the catalog and evaluated against it, not fitted to a
# score. (On the 50k catalog this evaluates to 2.996.)
PARTIAL_IDF_QUANTILE = 0.05
PARTIAL_COUNTS_AS_MATCH = 0.80   # coverage at which a partial counts for gating
# Proportional, not flat: a flat bonus would hand demoted (0.1x) override
# evidence its full value and break the §5.5 consistency asymmetry, and it
# would reward a bare material word as much as a 120-char feature string.
FIELD_WEIGHT_RATIO = 0.25        # of the constraint's own weight
FUZZY_ANCHOR_MIN = 0.5           # min token overlap to accept a fuzzy key
FUZZY_ANCHOR_KEYS = 3            # keys unioned into the pool (recall only)

# Recommendation gating (§5.4), sized from results_shadow_stage2.json: ranks
# plateau by turn 3 (rank ~1.05–1.15) and nothing leaves the top-10, so a
# rank-2+ turn-1 hit is a pure loss vs deferring. Depth 1 keeps genuine
# rank-1 hits (optimal) while blocking lock-ins.
GATE_INFORMED_FULL = 2     # leader must match ≥ this many active constraints…
GATE_FULL_MIN_TURN = 3     # …and it must be at least this turn, for full top-k
GATE_CAP_TURN = 5          # unconditional full top-k from this turn (hit-rate guard)
GATE_DEPTH = 1             # list length while uninformed

PROFILE_PRIOR_WEIGHT = 0.3
STALE_PENALTY = 0.75
MMR_DUP_PENALTY = 0.5


class Agent:
    """Deterministic exact-constraint agent (Stages 1–2 of plan.md)."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._products: dict[str, dict] = {}
        self._norm_text: dict[str, str] = {}
        self._card_text: dict[str, str] = {}   # title+features+details only
        self._token_text: dict[str, str] = {}  # space-delimited unique content tokens
        self._anchor_index: dict[str, list[str]] = {}
        self._anchor_tokens: dict[str, frozenset[str]] = {}
        self._doc_freq: dict[str, int] = {}
        self._n_docs: int = 0
        self._fallback_order: list[str] = []
        self._build_index()

    # ------------------------------------------------------------------ index

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        if not self.catalog_path.exists():
            return
        batch: list[tuple[str, ...]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self._products[parent_asin] = product
                parts = [str(product.get("title") or "")]
                for field_name in ("features", "details", "description", "categories"):
                    parts.extend(_flatten_field(product.get(field_name)))
                parts.append(str(product.get("store") or ""))
                self._norm_text[parent_asin] = normalize_text(" ".join(parts))
                # The evaluator's intent_card draws its constraint candidates
                # from `features` + `details` only, so a match inside those
                # fields is stronger evidence than one in a long description.
                card_parts = [str(product.get("title") or "")]
                for field_name in ("features", "details"):
                    card_parts.extend(_flatten_field(product.get(field_name)))
                self._card_text[parent_asin] = normalize_text(" ".join(card_parts))
                # Token-delimited view for `_coverage`. It must be built with
                # the same tokenizer the constraint side uses: `_terms` strips
                # the `%` that the normal form deliberately keeps inside
                # numbers, so testing " 100 " against the raw normal form
                # ("100% cotton") would never match.
                self._token_text[parent_asin] = " %s " % " ".join(
                    sorted(set(_terms(self._norm_text[parent_asin])))
                )
                anchor_key = normalize_text(coarse_category(product.get("categories") or []))
                self._anchor_index.setdefault(anchor_key, []).append(parent_asin)
                batch.append((
                    parent_asin,
                    str(product.get("title") or ""),
                    " ".join(_flatten_field(product.get("categories"))),
                    " ".join(_flatten_field(product.get("features"))),
                    " ".join(_flatten_field(product.get("details"))),
                    str(product.get("store") or ""),
                    " ".join(_flatten_field(product.get("description"))),
                ))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        self._n_docs = len(self._products)
        # `_norm_text` is already in the shared normal form, so a plain split
        # matches `_terms` here and keeps the one-time index build cheap.
        doc_freq = self._doc_freq
        for text in self._norm_text.values():
            for token in set(text.split()):
                if len(token) > 1 and token not in STOPWORDS:
                    doc_freq[token] = doc_freq.get(token, 0) + 1
        self._anchor_tokens = {
            key: frozenset(_terms(key)) for key in self._anchor_index
        }
        self._fallback_order = sorted(
            self._products,
            key=lambda parent_asin: (
                -float(self._products[parent_asin].get("average_rating") or 0.0),
                -float(self._products[parent_asin].get("rating_number") or 0.0),
                parent_asin,
            ),
        )

    @property
    def _partial_min_idf(self) -> float:
        """IDF of a token at `PARTIAL_IDF_QUANTILE` document frequency in *this*
        catalog. Scales with catalog size instead of assuming 50k."""
        return math.log(
            (self._n_docs + 1.0) / (PARTIAL_IDF_QUANTILE * self._n_docs + 1.0)
        )

    def _idf(self, token: str) -> float:
        """Smoothed inverse document frequency over the frozen catalog."""
        return math.log((self._n_docs + 1.0) / (self._doc_freq.get(token, 0) + 1.0))

    def _coverage(self, norm: str, token_text: str) -> float:
        """IDF-weighted fraction of a constraint's content tokens present in
        `token_text` (a product's `_token_text` entry). This is the graceful-
        degradation path for a payload that has been reworded: 'Rubber sole' -> 'soles made of Rubber' keeps 'rubber',
        which the whole-string substring test throws away entirely."""
        tokens = set(_terms(norm))
        if not tokens:
            return 0.0
        total = 0.0
        hit = 0.0
        for token in tokens:
            weight = self._idf(token)
            total += weight
            if f" {token} " in token_text:
                hit += weight
        if FEATURE_PARTIAL_IDF_FLOOR and hit < self._partial_min_idf:
            return 0.0
        return hit / total if total else 0.0

    # -------------------------------------------------------------- interface

    def reset(self, session_id: str, user_profile: dict) -> None:
        state = SessionState(profile=dict(user_profile or {}))
        tags = state.profile.get("preference_tags")
        if isinstance(tags, list):
            state.profile_tags = tuple(
                normalize_text(str(tag)) for tag in tags if normalize_text(str(tag))
            )
        self._sessions[session_id] = state

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self._respond(session_id, user_message, turn, top_k)
        except Exception:
            return {
                "message": "Could you tell me one more must-have detail?",
                "ask_attribute": "other",
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionState(profile={})
            self._sessions[session_id] = state
        self._extract(state, user_message or "")
        if self._demo_guard(state, turn):
            return {
                "message": "Happy to help — what kind of product are you looking for?",
                "ask_attribute": "other",
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
        version = (
            len(state.constraints), len(state.loose_terms), len(state.category_terms),
            state.budget_point, state.anchor is not None, state.anchor_bonus,
            self._n_active(state), len(state.penalized), state.dissatisfied,
        )
        if state.cache_version == version and state.cache_ranking:
            ranked = state.cache_ranking
        else:
            pool = self._build_pool(state, top_k)
            ranked = self._score_pool(state, pool)
            state.cache_version = version
            state.cache_ranking = ranked
        limit = max(0, int(top_k))
        if FEATURE_GATING:
            limit = min(limit, self._gate_depth(state, turn, ranked, limit))
        if FEATURE_MMR and limit > 1 and self._n_active(state) == 0 and state.p_buy < 0.8:
            ranked = self._mmr_diversify(ranked, limit)
        recommendations = [{"parent_asin": parent_asin} for parent_asin in ranked[:limit]]
        state.stale_shown.update(item["parent_asin"] for item in recommendations)
        # Ask "other" until the card is drained (§5.2) — the boundary one-off
        # ("no preference for other") lands in `exhausted`, never stops the asks.
        ask_attribute = "other" if state.drained < 2 else None
        if ask_attribute is None:
            message = "Here are the closest matches I found."
        elif recommendations:
            message = "Here are my strongest matches so far. Is there another must-have detail I should factor in?"
        else:
            message = "I need one more detail to narrow this down. Is there another must-have detail I should factor in?"
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    # ----------------------------------------------------- gating & priors

    def _n_active(self, state: SessionState) -> int:
        return sum(
            1 for constraint in state.constraints
            if not constraint.is_budget and not constraint.demoted
        )

    def _demo_guard(self, state: SessionState, turn: int) -> bool:
        """§5.2 rule 5: a message with nothing extractable (a human typing "Hi")
        must not fuzzy-match nonsense. The evaluator never produces this shape."""
        return (
            turn <= 2
            and state.anchor is None
            and not state.constraints
            and not state.category_terms
            and state.budget_point is None
            and len(state.loose_terms) <= 2
        )

    def _gate_depth(self, state: SessionState, turn: int, ranked: list[str], limit: int) -> int:
        """§5.4: truncate the list while the ranking is uninformed. Sized from
        results_shadow_stage2.json — ranks plateau at ~1.05–1.15 by turn 3 and
        the target never leaves the top-10, so deferring a rank-2+ early hit is
        strictly better than locking it in; depth 1 still captures genuine
        rank-1 hits immediately. Escapes protect hit-rate unconditionally."""
        if turn >= GATE_CAP_TURN or state.drained >= 1 or state.dissatisfied:
            return limit
        if FEATURE_REGIME_ESCAPE and self._matching_failed(state, ranked):
            # Constraints exist but nothing matches them: the exact-substring
            # path has failed (a reworded payload), so `informed` will never
            # rise and deferring to GATE_CAP_TURN only burns turns. Impossible
            # on the clean set — a card constraint is by construction a
            # substring of the target, so the leader always matches ≥ 1.
            return limit
        informed = 1 if state.budget_point is not None else 0
        if ranked:
            informed += state.memo.get(ranked[0], (0, 0.0, 0))[2]
        if informed >= GATE_INFORMED_FULL and turn >= GATE_FULL_MIN_TURN:
            return limit
        if FEATURE_PBUY and state.p_buy >= 0.95 and informed >= GATE_INFORMED_FULL:
            # A fully confident, fully informed buyer: no reason to defer.
            return limit
        return GATE_DEPTH

    def _matching_failed(self, state: SessionState, ranked: list[str]) -> bool:
        """True when ≥2 active constraints are held but the ranked leader
        matches none of them — the signature of the paraphrase regime."""
        if self._n_active(state) < 2 or not ranked:
            return False
        return state.memo.get(ranked[0], (0, 0.0, 0))[2] == 0

    def _mmr_diversify(self, ranked: list[str], limit: int) -> list[str]:
        """§5.3: on uninformed full-width turns only, avoid near-duplicate
        leaders (same store + same title head). Returns a new list."""
        head: list[str] = []
        deferred: list[str] = []
        seen_keys: set[tuple[str, str]] = set()
        for parent_asin in ranked[: limit * 3]:
            product = self._products.get(parent_asin, {})
            title_head = " ".join(_terms(str(product.get("title") or ""))[:3])
            key = (str(product.get("store") or "").lower(), title_head)
            if key in seen_keys:
                deferred.append(parent_asin)
            else:
                seen_keys.add(key)
                head.append(parent_asin)
            if len(head) >= limit:
                break
        merged = head + [item for item in deferred if item not in head]
        rest = [item for item in ranked if item not in set(merged[: limit * 3])]
        return merged + rest

    # ------------------------------------------------------------- extraction

    def _extract(self, state: SessionState, message: str) -> None:
        text = message.strip()
        if not text:
            return
        lowered = text.lower()
        if _DISSATISFIED_RE.search(lowered):
            state.drained = 0  # resume asking "other" (§5.5)
            self._strategy_switch(state)
            return
        if _NO_ADDITIONAL_RE.search(lowered):
            state.drained += 1
            return
        if _NO_PREF_RE.search(lowered):
            attribute = self._mentioned_attribute(lowered)
            if attribute:
                state.exhausted.add(attribute)
            return
        if state.anchor is None and self._try_opening(state, text):
            state.drained = 0
            return
        if _OVERRIDE_TRIGGER_RE.search(lowered[:20]):
            self._apply_override(state, text)
            return
        payload = self._reply_payload(text)
        if payload is not None:
            state.drained = 0
            if state.anchor is None:
                # Opening whose category defeated the anchor scan: keep the
                # pre-payload words as loose retrieval terms.
                for term in _terms(text[: len(text) - len(payload)]):
                    if term not in state.loose_terms:
                        state.loose_terms.append(term)
            for segment in self._split_payload(payload):
                self._add_constraint(state, segment)
            return
        # Unrecognized shape (paraphrase safety net): keep loose terms for retrieval only.
        state.drained = 0
        for term in _terms(text):
            if term not in state.loose_terms:
                state.loose_terms.append(term)

    def _strategy_switch(self, state: SessionState) -> None:
        """§5.5: the dissatisfaction reply is the ONLY trigger for the stale-rec
        penalty (§1 fact 13) and the relaxation ladder."""
        state.dissatisfied = True
        if FEATURE_STALE_PENALTY:
            state.penalized = frozenset(state.stale_shown)
        if FEATURE_RELAXATION:
            active = [
                constraint for constraint in state.constraints
                if not constraint.is_budget and not constraint.demoted
            ]
            if len(active) >= 2:  # never relax the only piece of evidence
                weakest = min(active, key=lambda constraint: constraint.weight)
                weakest.demoted = True
                weakest.weight *= 0.1
        state.memo.clear()
        state.cache_version = None
        state.cache_ranking = []

    def _mentioned_attribute(self, lowered: str) -> str | None:
        for attribute, phrase in _ATTRIBUTE_WORDS:
            if re.search(rf"\b{re.escape(phrase)}\b", lowered):
                return attribute
        return None

    def _reply_payload(self, text: str) -> str | None:
        match = _PAYLOAD_RE.search(text)
        if match:
            return match.group(1).strip()
        # Last resort: a colon-introduced tail that carries the "; " list joiner.
        head, sep, rest = text.partition(": ")
        if sep and "; " in rest:
            return rest.strip()
        return None

    def _split_payload(self, payload: str) -> list[str]:
        payload = payload.strip()
        if payload.endswith("."):
            payload = payload[:-1]  # sentence suffix only — never inner periods
        segments: list[str] = []
        for segment in payload.split("; "):
            for piece in _SECONDARY_JOINER_RE.split(segment):
                piece = piece.strip()
                if piece:
                    segments.append(piece)
        return segments

    def _try_opening(self, state: SessionState, text: str) -> bool:
        """Token-suffix anchor scan: the category tail is a suffix of the lead-in
        clause under any verb rewording ("Help me track down {cat}")."""
        candidates: list[tuple[str, str]] = []
        for match in _BOUNDARY_SPLIT_RE.finditer(text):
            candidates.append((text[:match.start()], text[match.end():]))
        candidates.append((text.rstrip(" .?!"), ""))
        candidates.sort(key=lambda item: len(item[0]))
        for head, tail in candidates:
            key = self._anchor_suffix(head)
            if key is None:
                continue
            self._install_anchor(state, key, ANCHOR_BONUS)
            self._apply_opening_tail(state, tail.strip())
            return True
        if FEATURE_INFIX_ANCHOR:
            # Exact key on any contiguous token window, longest first. Runs
            # only after the suffix scan has failed, so clean-set behaviour is
            # untouched by construction.
            key = self._anchor_infix(candidates[-1][0])
            if key is not None:
                self._install_anchor(state, key, ANCHOR_BONUS)
                return True
        if FEATURE_FUZZY_ANCHOR:
            # No exact category key: the tail was reworded. Union the closest
            # keys into the candidate pool, but award NO anchor bonus — the
            # best-overlap key is the wrong set often enough that boosting it
            # is worse than having no anchor at all. Never fires on the clean
            # set, where the opening carries the catalog's own category string
            # verbatim.
            head, tail = candidates[-1]
            keys = self._fuzzy_anchor_keys(head)
            if keys:
                pool: list[str] = []
                for key in keys:
                    pool.extend(self._anchor_index[key])
                    for term in _terms(key):
                        if term not in state.category_terms:
                            state.category_terms.append(term)
                state.anchor = tuple(dict.fromkeys(pool))
                state.anchor_set = frozenset()   # generation only, never scored
                state.anchor_bonus = 0.0
                self._apply_opening_tail(state, tail.strip())
                return True
        return False

    def _anchor_infix(self, text: str) -> str | None:
        """Longest contiguous token window of `text` that is an exact category
        key. Longest-first so 'rompers overalls jumpsuits' beats 'jumpsuits'."""
        tokens = normalize_text(text).split()
        for width in range(len(tokens), 0, -1):
            for start in range(0, len(tokens) - width + 1):
                key = " ".join(tokens[start:start + width])
                if key not in self._anchor_index:
                    continue
                if width == 1 and (len(key) <= 2 or key in STOPWORDS):
                    continue  # a lone short/function token is not a category
                return key
        return None

    def _install_anchor(self, state: SessionState, key: str, bonus: float) -> None:
        asins = self._anchor_index[key]
        state.anchor = tuple(asins)
        state.anchor_set = frozenset(asins)
        state.anchor_bonus = bonus
        for term in _terms(key):
            if term not in state.category_terms:
                state.category_terms.append(term)

    def _fuzzy_anchor_keys(self, head: str) -> list[str]:
        """Closest category keys by IDF-weighted token overlap. Recall only."""
        tokens = set(_terms(head))
        if not tokens:
            return []
        scored: list[tuple[float, str]] = []
        for key, key_tokens in self._anchor_tokens.items():
            shared = tokens & key_tokens
            if not shared:
                continue
            total = sum(self._idf(token) for token in key_tokens)
            if not total:
                continue
            score = sum(self._idf(token) for token in shared) / total
            if score >= FUZZY_ANCHOR_MIN:
                scored.append((score, key))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [key for _, key in scored[:FUZZY_ANCHOR_KEYS]]

    def _anchor_suffix(self, head: str) -> str | None:
        tokens = normalize_text(head).split()
        for start in range(len(tokens)):
            key = " ".join(tokens[start:])
            if key in self._anchor_index:
                if start == len(tokens) - 1 and (len(key) <= 2 or key in STOPWORDS):
                    continue  # single short/function token — not a credible category
                return key
        return None

    def _apply_opening_tail(self, state: SessionState, tail: str) -> None:
        if not tail:
            return
        lowered = tail.lower()
        payload = self._reply_payload(tail)
        if payload is not None:
            state.p_buy = 0.95  # a hard requirement disclosed at the opening (§5.1)
            for segment in self._split_payload(payload):
                self._add_constraint(state, segment)
        elif _EXPLORING_RE.search(lowered) or _NO_PREF_RE.search(lowered) or _NO_ADDITIONAL_RE.search(lowered):
            state.p_buy = 0.15  # explicitly still exploring (§5.1)
            return
        else:
            # intent_override opening: "I'm looking for {cat}. {old_value}" — the
            # old preference is still true of the target (§1 fact 10).
            value = tail[:-1] if tail.endswith(".") else tail
            self._add_constraint(state, value)

    def _apply_override(self, state: SessionState, text: str) -> None:
        """§5.5: demote (not delete) prior evidence; the asymmetric scoring of
        demoted constraints is the consistency gate."""
        for constraint in state.constraints:
            if not constraint.demoted:
                constraint.demoted = True
                if not constraint.is_budget:
                    constraint.weight *= 0.1
        state.budget_point = None  # old budget may be the overridden preference
        state.seen_keys = {key for key in state.seen_keys if not key.startswith("budget:")}
        state.memo.clear()  # demotion rewrites weights — incremental memo is stale
        state.cache_version = None
        state.cache_ranking = []
        state.stale_shown.clear()  # pre-override showings prove nothing (§1 fact 13)
        state.penalized = frozenset()
        state.drained = 0
        state.p_buy = 0.9  # they now know exactly what they want (§5.1)
        match = _OVERRIDE_VALUE_RE.search(text)
        if match:
            for segment in self._split_payload(match.group(1)):
                self._add_constraint(state, segment)

    def _add_constraint(self, state: SessionState, raw: str) -> None:
        raw = raw.strip()
        if not raw:
            return
        lowered = raw.lower()
        if FEATURE_BUDGET_GUARD:
            is_budget = len(raw) <= BUDGET_MAX_CHARS and bool(_BUDGET_PHRASE_RE.search(lowered))
        else:
            is_budget = bool(_BUDGET_WORD_RE.search(lowered) or re.search(r"\$\s*\d", lowered))
        point = extract_budget_point(lowered) if is_budget else None
        if is_budget and point is None:
            # Never drop the text: fall through and keep it as lexical evidence.
            is_budget = False
        if (
            not is_budget
            and len(raw) <= BUDGET_MAX_CHARS
            and _BUDGET_WORD_RE.search(lowered)
            and not any(character.isdigit() for character in lowered)
        ):
            # A short, numberless budget disclosure ("budget around fifty
            # dollars"): no price filter is possible and its words are pure
            # noise to the lexical matcher, so it is dropped outright. Long
            # strings are never dropped — that is the misrouting this guards.
            return
        if is_budget:
            key = f"budget:{point}"
            if key in state.seen_keys:
                return
            state.seen_keys.add(key)
            state.constraints.append(_Constraint(raw, "", 0.0, True, point))
            state.budget_point = point
            return
        body = re.sub(r"^color:\s*", "", raw, flags=re.IGNORECASE)
        norm = normalize_text(body)
        if not norm:
            return
        weight = 0.8 + 0.55 * min(len(norm.split()), 12)
        if norm in state.seen_keys:
            # Re-revealed after an override (the simulator can re-reveal the
            # "ignored" value — §1 fact 10): promote back to full weight.
            promoted = False
            for constraint in state.constraints:
                if constraint.norm == norm and constraint.demoted:
                    constraint.demoted = False
                    constraint.weight = weight
                    promoted = True
            if promoted:
                state.memo.clear()
                state.cache_version = None
            return
        state.seen_keys.add(norm)
        state.constraints.append(_Constraint(raw, norm, weight, False, None))
        state.p_buy = min(0.95, state.p_buy + 0.15)  # constraint mass raises confidence

    # -------------------------------------------------------------- retrieval

    def _build_pool(self, state: SessionState, top_k: int) -> list[str]:
        pool: list[str] = []
        seen: set[str] = set()
        if state.anchor:
            pool.extend(state.anchor)
            seen.update(state.anchor)
        if FEATURE_MULTI_QUERY:
            groups: list[list[str]] = [
                _terms(constraint.verbatim) for constraint in state.constraints
                if not constraint.is_budget
            ]
            groups.append(list(state.category_terms) + list(state.loose_terms))
            for group in groups:
                terms = list(dict.fromkeys(group))[:20]
                if not terms:
                    continue
                expression = " OR ".join(f'"{term}"' for term in terms)
                for parent_asin in self._fts_search(expression, MULTI_QUERY_LIMIT):
                    if parent_asin not in seen:
                        seen.add(parent_asin)
                        pool.append(parent_asin)
        else:
            query_terms: list[str] = []
            for constraint in state.constraints:
                query_terms.extend(_terms(constraint.verbatim))
            query_terms.extend(state.category_terms)
            query_terms.extend(state.loose_terms)
            deduped = list(dict.fromkeys(query_terms))[:40]
            if deduped:
                expression = " OR ".join(f'"{term}"' for term in deduped)
                for parent_asin in self._fts_search(expression, 300):
                    if parent_asin not in seen:
                        seen.add(parent_asin)
                        pool.append(parent_asin)
        if len(pool) < max(30, top_k * 3):
            for parent_asin in self._fallback_order:
                if parent_asin not in seen:
                    seen.add(parent_asin)
                    pool.append(parent_asin)
                if len(pool) >= 300:
                    break
        return pool

    def _fts_search(self, expression: str, limit: int) -> list[str]:
        if not expression:
            return []
        try:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(row[0]) for row in rows]

    # ---------------------------------------------------------------- scoring

    def _score_pool(self, state: SessionState, pool: Iterable[str]) -> list[str]:
        constraints = state.constraints
        n_scorable = sum(1 for constraint in constraints if not constraint.is_budget and not constraint.demoted)
        scored: list[tuple[float, str]] = []
        for parent_asin in pool:
            product = self._products.get(parent_asin)
            if product is None:
                continue
            entry = state.memo.get(parent_asin)
            if entry is None:
                entry = [0, 0.0, 0]
                state.memo[parent_asin] = entry
            if entry[0] < len(constraints):
                text = self._norm_text.get(parent_asin, "")
                card_text = self._card_text.get(parent_asin, "")
                token_text = self._token_text.get(parent_asin, " ")
                for constraint in constraints[entry[0]:]:
                    if constraint.is_budget:
                        continue
                    if constraint.norm in text:
                        # Demoted evidence adds its (already 0.1×) weight when
                        # matched but never penalizes when absent — that
                        # asymmetry is the §5.5 consistency gate.
                        entry[1] += constraint.weight
                        if FEATURE_FIELD_WEIGHT and constraint.norm in card_text:
                            entry[1] += constraint.weight * FIELD_WEIGHT_RATIO
                        if not constraint.demoted:
                            entry[2] += 1
                        continue
                    coverage = (
                        self._coverage(constraint.norm, token_text)
                        if FEATURE_PARTIAL_MATCH else 0.0
                    )
                    if coverage >= PARTIAL_MIN:
                        # Sub-additive in coverage: a full substring match always
                        # outscores any partial one, so clean-set ordering among
                        # exact matches is untouched.
                        entry[1] += constraint.weight * PARTIAL_SCALE * coverage * coverage
                        if not constraint.demoted:
                            entry[1] -= UNMATCHED_PENALTY * (1.0 - coverage)
                            if coverage >= PARTIAL_COUNTS_AS_MATCH:
                                entry[2] += 1
                    elif not constraint.demoted:
                        entry[1] -= UNMATCHED_PENALTY
                entry[0] = len(constraints)
            score = entry[1]
            if n_scorable and entry[2] == n_scorable:
                score += ALL_MATCHED_BONUS
            score += self._budget_score(state, product)
            score += 0.08 * float(product.get("average_rating") or 0.0)
            score += 0.05 * math.log1p(float(product.get("rating_number") or 0.0))
            if parent_asin in state.anchor_set:
                score += state.anchor_bonus or ANCHOR_BONUS
            if (
                FEATURE_PROFILE_PRIOR and not n_scorable
                and state.budget_point is None and state.profile_tags
            ):
                # Zero-evidence turns only: elsewhere constraints dominate and
                # the aggregate profile is noise (§5.1 prior).
                text = self._norm_text.get(parent_asin, "")
                padded = f" {text} "
                score += PROFILE_PRIOR_WEIGHT * sum(
                    1 for tag in state.profile_tags if f" {tag} " in padded
                )
            if FEATURE_STALE_PENALTY and state.penalized and parent_asin in state.penalized:
                score -= STALE_PENALTY  # only ever set on dissatisfaction (§5.5)
            scored.append((score, parent_asin))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [parent_asin for _, parent_asin in scored]

    def _budget_score(self, state: SessionState, product: dict) -> float:
        point = state.budget_point
        if point is None:
            return 0.0
        try:
            price = float(product.get("price"))
        except (TypeError, ValueError):
            return -BUDGET_NO_PRICE_PENALTY
        if abs(price - point) <= max(0.01 * point, 0.01):
            return BUDGET_EXACT_BONUS
        if 0.65 * point <= price <= 1.35 * point:
            return BUDGET_RANGE_BONUS
        return -BUDGET_MISS_PENALTY
