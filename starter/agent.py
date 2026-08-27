from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
PRICE_RE = re.compile(r"(?:\$|usd\s*)?(\d+(?:\.\d+)?)", re.IGNORECASE)
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}
STOPWORDS = {
    "a", "about", "actually", "additional", "an", "and", "are", "as", "ask",
    "at", "be", "but", "by", "closest", "do", "don", "earlier", "exploring", "for", "from",
    "have", "here", "i", "in", "is", "it", "judgment", "key", "looking",
    "ignore", "matches", "me", "my", "need", "not", "of", "on", "one", "options",
    "or", "please", "preference", "quite", "requirement", "right", "some",
    "specific", "that", "the", "this", "those", "to", "use", "want",
    "what", "with", "would", "you", "your",
}
GENERIC_CATEGORIES = {
    "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry",
    "shoes", "jewelry", "women", "men", "girls", "boys", "novelty",
}
MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "fleece", "denim", "suede", "mesh", "canvas",
    "rubber", "satin", "linen", "lace", "acrylic",
}
COLORS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "gold", "silver", "beige",
    "tan", "navy", "ivory", "multicolor", "clear",
}
SIZES = {
    "xs", "small", "medium", "large", "xl", "xxl", "wide", "narrow",
    "petite", "plus", "tall", "short", "regular", "slim",
}
STYLE_WORDS = {
    "casual", "formal", "classic", "modern", "vintage", "athletic",
    "sport", "dress", "western", "boho", "lace", "sleeveless", "hooded",
    "crew", "vneck", "v", "neck", "skinny", "straight", "relaxed",
    "compression", "loose", "fitted", "department", "fit", "style",
}
USE_CASE_WORDS = {
    "running", "hiking", "walking", "work", "workout", "gym", "winter",
    "summer", "outdoor", "travel", "wedding", "party", "rain", "snow",
    "beach", "school", "dance", "yoga", "cycling", "training", "warmth",
    "weather", "comfort", "durability",
}
PRODUCT_NOUNS = {
    "bra", "boot", "boots", "bracelet", "cap", "coat", "dress", "earrings",
    "gloves", "hat", "hoodie", "jacket", "jeans", "leggings", "necklace",
    "pajamas", "pants", "ring", "sandals", "shirt", "shoe", "shoes",
    "shorts", "skirt", "sneakers", "socks", "sweater", "swimsuit", "tee",
    "tights", "top", "wallet", "watch",
}
ASK_PRIORITY = [
    "material", "color", "feature", "style", "use_case",
    "budget", "size", "brand", "other", "category",
]


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _slot_map() -> dict[str, set[str]]:
    return {attribute: set() for attribute in ALLOWED_ATTRIBUTES}


@dataclass
class SessionState:
    profile: dict
    history: list[str] = field(default_factory=list)
    slots: dict[str, set[str]] = field(default_factory=_slot_map)
    category_terms: set[str] = field(default_factory=set)
    exclusions: set[str] = field(default_factory=set)
    no_preference: set[str] = field(default_factory=set)
    asked: list[str] = field(default_factory=list)
    last_asked: str | None = None
    profile_terms: set[str] = field(default_factory=set)
    budget_min: float | None = None
    budget_max: float | None = None


class Agent:
    """Offline conversational search prototype with deterministic state and reranking."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._products: dict[str, dict] = {}
        self._product_text: dict[str, str] = {}
        self._product_terms: dict[str, set[str]] = {}
        self._fallback_order: list[str] = []
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        if not self.catalog_path.exists():
            return

        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                features = _text(product.get("features"))
                details = _text(product.get("details"))
                store = _text(product.get("store"))
                description = _text(product.get("description"))
                search_text = " ".join([title, categories, features, details, store, description])

                self._products[parent_asin] = product
                self._product_text[parent_asin] = search_text.lower()
                self._product_terms[parent_asin] = set(_terms(search_text))
                batch.append((parent_asin, title, categories, features, details, store, description))
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

    def reset(self, session_id: str, user_profile: dict) -> None:
        state = SessionState(profile=dict(user_profile or {}))
        profile_text = " ".join([
            _text(state.profile.get("summary")),
            _text(state.profile.get("preference_tags")),
        ])
        state.profile_terms = set(_terms(profile_text))
        self._sessions[session_id] = state

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        state = self._sessions[session_id]
        ignore_latest = self._is_no_preference(user_message or "")
        self._observe(state, user_message)
        retrieval_message = "" if ignore_latest else user_message
        pool = self._retrieve_pool(state, retrieval_message, max(60, top_k * 30))
        ranked = self._rerank(state, retrieval_message, pool)
        recommendations = [{"parent_asin": parent_asin} for parent_asin in ranked[:top_k]]
        ask_attribute = self._choose_attribute(state, ranked[:80], turn)
        if ask_attribute is not None:
            state.asked.append(ask_attribute)
            state.last_asked = ask_attribute
        message = self._message_for(ask_attribute, bool(recommendations))
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _observe(self, state: SessionState, user_message: str) -> None:
        text = user_message or ""
        lowered = text.lower()
        state.history.append(text)

        override = bool(re.search(r"\b(actually|instead|forget|ignore|replace)\b", lowered))
        if override:
            preserved_category = set(state.slots["category"])
            state.slots = _slot_map()
            state.slots["category"].update(preserved_category)
            state.no_preference.clear()
            state.budget_min = None
            state.budget_max = None

        no_preference = self._is_no_preference(lowered)
        if no_preference:
            attribute = self._mentioned_attribute(lowered) or state.last_asked
            if attribute in ALLOWED_ATTRIBUTES:
                state.no_preference.add(attribute)
                if attribute not in {"category", "other"}:
                    state.slots[attribute].clear()
            return

        category_phrase = self._looking_for_phrase(text)
        if category_phrase:
            for term in _terms(category_phrase):
                if term in PRODUCT_NOUNS or term not in GENERIC_CATEGORIES:
                    state.category_terms.add(term)
                    state.slots["category"].add(term)

        for phrase in self._constraint_phrases(text):
            self._add_phrase(state, phrase)
        self._extract_surface_slots(state, text)
        self._extract_exclusions(state, lowered)

    def _mentioned_attribute(self, lowered: str) -> str | None:
        for attribute in ALLOWED_ATTRIBUTES:
            if re.search(rf"\b{re.escape(attribute.replace('_', ' '))}\b", lowered):
                return attribute
        return None

    def _is_no_preference(self, text: str) -> bool:
        lowered = text.lower()
        return bool(
            re.search(r"\b(no|don't|do not)\s+(?:have\s+)?(?:an?\s+)?(?:additional\s+)?preference\b", lowered)
            or "use your judgment" in lowered
        )

    def _looking_for_phrase(self, text: str) -> str:
        match = re.search(r"\blooking for\s+(.+?)(?:[.;]|\s+but\b|\s+and\b|$)", text, re.IGNORECASE)
        return self._clean_phrase(match.group(1)) if match else ""

    def _constraint_phrases(self, text: str) -> list[str]:
        phrases: list[str] = []
        for pattern in (
            r"(?:key requirement is|what matters is|what i need is|requirement is):?\s*(.+)",
            r"\b(?:need|want|prefer|prioritize)\s+(.+)",
        ):
            for match in re.finditer(pattern, text, re.IGNORECASE):
                tail = match.group(1)
                pieces = re.split(r";|, and |\.| but ", tail)
                phrases.extend(self._clean_phrase(piece) for piece in pieces)
        return [phrase for phrase in phrases if phrase]

    def _clean_phrase(self, phrase: str) -> str:
        phrase = re.sub(r"\s+", " ", phrase)
        return phrase.strip(" -:;,.\t\n")[:120].rstrip()

    def _add_phrase(self, state: SessionState, phrase: str) -> None:
        attribute = self._classify_phrase(phrase)
        state.slots[attribute].add(phrase.lower())
        if attribute == "category":
            state.category_terms.update(_terms(phrase))
        if attribute == "budget":
            self._parse_budget(state, phrase)

    def _classify_phrase(self, phrase: str) -> str:
        lowered = phrase.lower()
        terms = set(_terms(lowered))
        if "budget" in lowered or "$" in lowered or re.search(r"\b(under|below|less than|around|over)\b", lowered):
            return "budget"
        if terms & MATERIALS:
            return "material"
        if terms & COLORS or "color" in lowered:
            return "color"
        if terms & SIZES or re.search(r"\bsize\s+[0-9a-z.]+\b", lowered):
            return "size"
        if "brand" in lowered or "store" in lowered:
            return "brand"
        if terms & USE_CASE_WORDS:
            return "use_case"
        if terms & STYLE_WORDS:
            return "style"
        if terms & PRODUCT_NOUNS:
            return "category"
        return "feature"

    def _extract_surface_slots(self, state: SessionState, text: str) -> None:
        lowered = text.lower()
        terms = set(_terms(lowered))
        for material in sorted(terms & MATERIALS):
            state.slots["material"].add(material)
        for color in sorted(terms & COLORS):
            state.slots["color"].add(color)
        for size in sorted(terms & SIZES):
            state.slots["size"].add(size)
        for use_case in sorted(terms & USE_CASE_WORDS):
            state.slots["use_case"].add(use_case)
        for style in sorted(terms & STYLE_WORDS):
            state.slots["style"].add(style)
        for noun in sorted(terms & PRODUCT_NOUNS):
            state.category_terms.add(noun)
            state.slots["category"].add(noun)
        self._parse_budget(state, text)

    def _parse_budget(self, state: SessionState, text: str) -> None:
        lowered = text.lower()
        numbers = [float(match.group(1)) for match in PRICE_RE.finditer(lowered)]
        if not numbers or not any(marker in lowered for marker in ("$", "budget", "under", "below", "less", "around", "over")):
            return
        if re.search(r"\b(under|below|less than|max|maximum|<=)\b", lowered):
            state.budget_max = min(numbers)
        elif re.search(r"\b(over|above|more than|at least|>=)\b", lowered):
            state.budget_min = max(numbers)
        elif "between" in lowered and len(numbers) >= 2:
            state.budget_min, state.budget_max = min(numbers[:2]), max(numbers[:2])
        else:
            target = numbers[0]
            state.budget_min = max(0.0, target * 0.65)
            state.budget_max = target * 1.35
        state.slots["budget"].add(self._clean_phrase(text).lower())

    def _extract_exclusions(self, state: SessionState, lowered: str) -> None:
        if "preference" in lowered:
            return
        for match in re.finditer(r"\b(?:avoid|without|no|not|don't want|do not want)\s+([a-z0-9 $.-]+)", lowered):
            for term in _terms(match.group(1))[:5]:
                if term not in {"preference", "additional"}:
                    state.exclusions.add(term)

    def _retrieve_pool(self, state: SessionState, user_message: str, limit: int) -> list[str]:
        query_terms = self._query_terms(state, user_message)
        expressions: list[str] = []
        if len(query_terms) >= 2:
            expressions.append(" ".join(f'"{term}"' for term in query_terms[:12]))
        if query_terms:
            expressions.append(" OR ".join(f'"{term}"' for term in query_terms[:40]))
        if state.category_terms:
            expressions.append(" OR ".join(f'"{term}"' for term in sorted(state.category_terms)[:20]))

        pool: list[str] = []
        for expression in expressions:
            for parent_asin in self._fts_search(expression, limit):
                if parent_asin not in pool:
                    pool.append(parent_asin)
                if len(pool) >= limit:
                    break
            if len(pool) >= max(20, limit // 3):
                break
        if len(pool) < min(limit, 20):
            for parent_asin in self._fallback_order:
                if parent_asin not in pool:
                    pool.append(parent_asin)
                if len(pool) >= limit:
                    break
        return pool

    def _query_terms(self, state: SessionState, user_message: str) -> list[str]:
        values: list[str] = []
        for attribute, slot_values in state.slots.items():
            if attribute in state.no_preference:
                continue
            values.extend(slot_values)
        values.extend(state.category_terms)
        values.append(user_message)
        if state.profile_terms:
            values.extend(sorted(state.profile_terms)[:8])
        terms = _terms(" ".join(values))
        return _dedupe(term for term in terms if term not in state.exclusions)

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

    def _rerank(self, state: SessionState, user_message: str, pool: list[str]) -> list[str]:
        latest_terms = set(_terms(user_message))
        scored: list[tuple[float, str]] = []
        total = max(1, len(pool))
        for rank, parent_asin in enumerate(pool):
            product = self._products.get(parent_asin)
            if not product:
                continue
            text = self._product_text.get(parent_asin, "")
            terms = self._product_terms.get(parent_asin, set())
            score = 1.0 - (rank / total)
            score += self._slot_score(state, text, terms, product)
            score += 0.18 * len(latest_terms & terms)
            score += 0.08 * float(product.get("average_rating") or 0.0)
            score += 0.05 * math.log1p(float(product.get("rating_number") or 0.0))
            scored.append((score, parent_asin))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [parent_asin for _, parent_asin in scored]

    def _slot_score(self, state: SessionState, text: str, terms: set[str], product: dict) -> float:
        score = 0.0
        for term in state.category_terms:
            score += 1.4 if term in terms else -0.2
        for attribute, values in state.slots.items():
            if attribute in {"budget", "category"} or attribute in state.no_preference:
                continue
            for value in values:
                value_terms = set(_terms(value))
                if not value_terms:
                    continue
                matched = len(value_terms & terms)
                if value.lower() in text:
                    score += 2.4
                elif matched:
                    score += 0.7 * matched / len(value_terms)
                else:
                    score -= 0.45 if attribute in {"feature", "style", "use_case"} else 0.8
        for term in state.exclusions:
            if term in terms:
                score -= 3.5
        score += self._budget_score(state, product)
        return score

    def _budget_score(self, state: SessionState, product: dict) -> float:
        if state.budget_min is None and state.budget_max is None:
            return 0.0
        try:
            price = float(product.get("price"))
        except (TypeError, ValueError):
            return -0.15
        score = 0.0
        if state.budget_min is not None:
            score += 1.1 if price >= state.budget_min else -1.2
        if state.budget_max is not None:
            if price <= state.budget_max:
                score += 1.6
            else:
                overage = (price - state.budget_max) / max(state.budget_max, 1.0)
                score -= min(4.0, 1.5 + overage * 2.0)
        return score

    def _choose_attribute(self, state: SessionState, ranked_pool: list[str], turn: int) -> str | None:
        if turn >= 10:
            return None
        filled = {attribute for attribute, values in state.slots.items() if values}
        if len(filled - {"category"}) >= 4 and turn >= 4:
            return None

        diversity = self._attribute_diversity(ranked_pool)
        best_attribute: str | None = None
        best_score = -1.0
        for index, attribute in enumerate(ASK_PRIORITY):
            if attribute in state.no_preference or attribute in filled:
                continue
            repeat_penalty = 1.5 if attribute in state.asked else 0.0
            value_count = diversity.get(attribute, 0)
            if attribute == "other":
                value_count = max(value_count, 2 if len(state.asked) >= 3 else 0)
            if value_count < 2 and attribute not in {"feature", "other"}:
                continue
            priority = (len(ASK_PRIORITY) - index) / len(ASK_PRIORITY)
            profile_bonus = 0.5 if attribute in state.profile_terms else 0.0
            score = priority + min(value_count, 8) * 0.18 + profile_bonus - repeat_penalty
            if score > best_score:
                best_score = score
                best_attribute = attribute
        return best_attribute

    def _attribute_diversity(self, ranked_pool: list[str]) -> dict[str, int]:
        values: dict[str, set[str]] = {attribute: set() for attribute in ALLOWED_ATTRIBUTES}
        for parent_asin in ranked_pool[:80]:
            product = self._products.get(parent_asin, {})
            text = self._product_text.get(parent_asin, "")
            terms = self._product_terms.get(parent_asin, set())
            values["material"].update(sorted(terms & MATERIALS)[:3])
            values["color"].update(sorted(terms & COLORS)[:3])
            values["size"].update(sorted(terms & SIZES)[:3])
            values["style"].update(sorted(terms & STYLE_WORDS)[:3])
            values["use_case"].update(sorted(terms & USE_CASE_WORDS)[:3])
            values["brand"].add(str(product.get("store") or "").lower()[:40])
            values["category"].update(
                category.lower()
                for category in (product.get("categories") or [])[-2:]
                if str(category).lower() not in GENERIC_CATEGORIES
            )
            try:
                price = float(product.get("price"))
            except (TypeError, ValueError):
                price = 0.0
            if price:
                values["budget"].add(str(int(price // 25) * 25))
            for feature in product.get("features") or []:
                for term in _terms(str(feature))[:4]:
                    if term not in MATERIALS and term not in COLORS:
                        values["feature"].add(term)
            if not values["feature"]:
                values["feature"].update(term for term in _terms(text)[:8] if term not in PRODUCT_NOUNS)
        values["other"].update(values["feature"])
        return {attribute: len({value for value in attribute_values if value}) for attribute, attribute_values in values.items()}

    def _message_for(self, ask_attribute: str | None, has_recommendations: bool) -> str:
        prefix = "I found a few strong candidates." if has_recommendations else "I need one more detail to narrow this down."
        questions = {
            "category": "What product category should I focus on?",
            "material": "Do you have a preferred material?",
            "color": "Is there a color you want me to prioritize?",
            "size": "Is there a size or fit detail I should use?",
            "style": "What style or fit would you prefer?",
            "brand": "Is there a brand or store you prefer?",
            "budget": "What budget range should I stay near?",
            "feature": "What feature matters most?",
            "use_case": "What will you mainly use it for?",
            "other": "Is there another must-have detail I should factor in?",
        }
        if ask_attribute is None:
            return "Here are the closest matches I found."
        return f"{prefix} {questions[ask_attribute]}"
