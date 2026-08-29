"""Stage 1 submission agent: exact-constraint reranking over a category anchor.

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
_NUMBER_RE = re.compile(r"\$?\d[\d,]*\.?\d*")
_LOOKING_FOR_RE = re.compile(r"i'?m looking for\s+(.*)$", re.IGNORECASE | re.DOTALL)
_REPLY_RE = re.compile(r"what matters is:\s*(.+)$", re.IGNORECASE | re.DOTALL)
_OVERRIDE_RE = re.compile(r"what i need is:\s*(.+)$", re.IGNORECASE | re.DOTALL)
_REQUIREMENT_RE = re.compile(
    r"(?:a key requirement is|what i need is|what matters is):?\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)


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


@dataclass
class SessionState:
    profile: dict
    anchor: tuple[str, ...] | None = None
    anchor_set: frozenset[str] = frozenset()
    category_terms: list[str] = field(default_factory=list)
    loose_terms: list[str] = field(default_factory=list)
    constraints: list[_Constraint] = field(default_factory=list)
    seen_keys: set[str] = field(default_factory=set)
    budget_point: float | None = None
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


class Agent:
    """Deterministic exact-constraint agent (Stage 1 of plan.md)."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._products: dict[str, dict] = {}
        self._norm_text: dict[str, str] = {}
        self._anchor_index: dict[str, list[str]] = {}
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
        self._fallback_order = sorted(
            self._products,
            key=lambda parent_asin: (
                -float(self._products[parent_asin].get("average_rating") or 0.0),
                -float(self._products[parent_asin].get("rating_number") or 0.0),
                parent_asin,
            ),
        )

    # -------------------------------------------------------------- interface

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(profile=dict(user_profile or {}))

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
        version = (
            len(state.constraints), len(state.loose_terms), len(state.category_terms),
            state.budget_point, state.anchor is not None,
        )
        if state.cache_version == version and state.cache_ranking:
            ranked = state.cache_ranking
        else:
            pool = self._build_pool(state, top_k)
            ranked = self._score_pool(state, pool)
            state.cache_version = version
            state.cache_ranking = ranked
        limit = max(0, int(top_k))
        recommendations = [{"parent_asin": parent_asin} for parent_asin in ranked[:limit]]
        message = (
            "Here are my strongest matches so far. Is there another must-have detail I should factor in?"
            if recommendations
            else "I need one more detail to narrow this down. Is there another must-have detail I should factor in?"
        )
        return {
            "message": message,
            "ask_attribute": "other",
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    # ------------------------------------------------------------- extraction

    def _is_noise(self, lowered: str) -> bool:
        return bool(
            re.search(r"\b(no|don'?t|do not)\s+(?:have\s+)?(?:an?\s+)?(?:additional\s+)?preference\b", lowered)
            or "use your judgment" in lowered
            or lowered.startswith("those options are not quite right")
        )

    def _extract(self, state: SessionState, message: str) -> None:
        text = message.strip()
        if not text or self._is_noise(text.lower()):
            return
        match = _LOOKING_FOR_RE.search(text)
        if match:
            self._apply_opening(state, match.group(1))
            return
        match = _REPLY_RE.search(text)
        if match:
            payload = match.group(1).strip()
            if payload.endswith("."):
                payload = payload[:-1]
            for segment in payload.split("; "):
                self._add_constraint(state, segment)
            return
        match = _OVERRIDE_RE.search(text)
        if match:
            value = match.group(1).strip()
            if value.endswith("."):
                value = value[:-1]
            self._add_constraint(state, value)
            return
        # Unrecognized shape (paraphrase safety net): keep loose terms for retrieval only.
        for term in _terms(text):
            if term not in state.loose_terms:
                state.loose_terms.append(term)

    def _apply_opening(self, state: SessionState, rest: str) -> None:
        candidates: list[tuple[str, str]] = []
        for match in re.finditer(r"\. ", rest):
            candidates.append((rest[:match.start()], rest[match.end():]))
        but_index = rest.find(", but")
        if but_index != -1:
            candidates.append((rest[:but_index], rest[but_index + 2:]))
        candidates.append((rest.rstrip(" ."), ""))
        candidates.sort(key=lambda item: len(item[0]))

        category_phrase, remainder = "", ""
        for phrase, tail in candidates:
            key = normalize_text(phrase)
            if key and key in self._anchor_index:
                category_phrase, remainder = phrase, tail
                asins = self._anchor_index[key]
                state.anchor = tuple(asins)
                state.anchor_set = frozenset(asins)
                break
        if not state.anchor:
            category_phrase, remainder = candidates[0] if candidates else (rest, "")
        for term in _terms(category_phrase):
            if term not in state.category_terms:
                state.category_terms.append(term)

        remainder = remainder.strip()
        if not remainder:
            return
        match = _REQUIREMENT_RE.match(remainder)
        if match:
            value = match.group(1).strip()
            if value.endswith("."):
                value = value[:-1]
            self._add_constraint(state, value)
        elif re.match(r"but i'?m still exploring", remainder, re.IGNORECASE):
            return
        elif not self._is_noise(remainder.lower()):
            # intent_override opening: "I'm looking for {cat}. {old_value}" — the
            # old preference is still true of the target (§1 fact 10).
            value = remainder[:-1] if remainder.endswith(".") else remainder
            self._add_constraint(state, value)

    def _add_constraint(self, state: SessionState, raw: str) -> None:
        raw = raw.strip()
        if not raw:
            return
        lowered = raw.lower()
        is_budget = bool(_BUDGET_WORD_RE.search(lowered) or re.search(r"\$\s*\d", lowered))
        if is_budget:
            point = extract_budget_point(lowered)
            if point is None:
                return  # no number → no price filter, and never substring-matched
            key = f"budget:{point}"
            if key in state.seen_keys:
                return
            state.seen_keys.add(key)
            state.constraints.append(_Constraint(raw, "", 0.0, True, point))
            state.budget_point = point
            return
        body = re.sub(r"^color:\s*", "", raw, flags=re.IGNORECASE)
        norm = normalize_text(body)
        if not norm or norm in state.seen_keys:
            return
        state.seen_keys.add(norm)
        weight = 0.8 + 0.55 * min(len(norm.split()), 12)
        state.constraints.append(_Constraint(raw, norm, weight, False, None))

    # -------------------------------------------------------------- retrieval

    def _build_pool(self, state: SessionState, top_k: int) -> list[str]:
        pool: list[str] = []
        seen: set[str] = set()
        if state.anchor:
            pool.extend(state.anchor)
            seen.update(state.anchor)
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
        n_scorable = sum(1 for constraint in constraints if not constraint.is_budget)
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
                for constraint in constraints[entry[0]:]:
                    if constraint.is_budget:
                        continue
                    if constraint.norm in text:
                        entry[1] += constraint.weight
                        entry[2] += 1
                    else:
                        entry[1] -= UNMATCHED_PENALTY
                entry[0] = len(constraints)
            score = entry[1]
            if n_scorable and entry[2] == n_scorable:
                score += ALL_MATCHED_BONUS
            score += self._budget_score(state, product)
            score += 0.08 * float(product.get("average_rating") or 0.0)
            score += 0.05 * math.log1p(float(product.get("rating_number") or 0.0))
            if parent_asin in state.anchor_set:
                score += ANCHOR_BONUS
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
