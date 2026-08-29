"""L3: meaning-preserving *semantic* rewrites of constraint payloads.

Why this exists (REPORT §6.1): L2's payload mutations are no-ops against this
agent by construction — `_flip_first_letter` is erased by the agent's
lowercasing normal form, and the budget rephrasing is bypassed because budget
constraints are regex-extracted and never substring-matched. So the 0.959 L2
row measures frame parsing and payload splitting, not lexical matching of
payloads. L3 is the missing test: every rewrite keeps the meaning (a human
reading it identifies the same product) but breaks the exact-substring path.

Two levels, deliberately split:

- **L3a** — payloads are rewritten; the category tail is left byte-identical,
  so the anchor still resolves. This is the realistic "organizer added a
  paraphraser to the customer simulator" scenario.
- **L3b** — the category tail is rewritten too, so the anchor scan fails as
  well. This is a floor, not a forecast: it models an evaluator whose category
  string no longer matches the catalog's own `categories` field.
- **catdrift** — the category tail is rewritten and *nothing else is*. REPORT
  §6.2 names spec-faithful card construction as the load-bearing assumption and
  category replication as the part of it most likely to drift; L3b never
  isolates that, because the same rewriter also rewrites the payloads, so the
  category words that appear *inside* a payload drift in lockstep with the
  opening's. Holding payloads byte-identical is what exposes the interaction:
  the anchor scan runs on every turn while no anchor is held, and it is then
  reading verbatim catalog text that can contain a category name.

Rewrite rules are grounded in the actual public-set constraint distribution
(800 instances, 342 distinct), which is dominated by a small set of templated
Amazon metadata shapes: `<X> sole`, `<X> closure`, `N% <Material>`,
`Machine Wash`, `Hand Wash Only`, `Imported`, `color: <c>`, `<X> lining`.

**Single-word constraints are left unchanged by default.** 276 of 800 (34.5%)
are a bare material or origin word (`cotton`, `polyester`, `Imported`). A
meaning-preserving rewording keeps the head noun, so there is nothing to
rewrite that a reader would still call the same constraint. They are the floor
under the measured degradation. `--mutate-singles` wraps them anyway
("cotton" -> "made of cotton") to measure the harsher variant, in which the
whole-payload substring test fails even though the head noun survives.

CAVEAT, carried deliberately: this rewriter shares authorship with the agent,
which is exactly the criticism §6.1 makes of L1/L2. It is directional
evidence, not a guarantee.
"""
from __future__ import annotations

import random
import re

from tools.paraphraser import (
    BUDGET_CONSTRAINT_RE,
    BUDGET_VARIANTS,
    L1_TEMPLATES,
    L2_EXTRA_TEMPLATES,
    L2_JOINERS,
    Paraphraser,
)

L3_LEVELS = ("L3a", "L3b", "catdrift")
MUTATION_RATE = 0.6

_MATERIAL_WORDS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "synthetic", "suede", "denim", "linen", "cashmere",
}

# ---------------------------------------------------------------- structural

_COLOR_RE = re.compile(r"^color:\s*(?P<c>[a-z]+)$", re.IGNORECASE)
_SOLE_RE = re.compile(r"^(?P<x>.+?)\s+sole$", re.IGNORECASE)
_CLOSURE_RE = re.compile(r"^(?:closure:\s*)?(?P<x>.+?)\s+closure$", re.IGNORECASE)
_LINING_RE = re.compile(r"^(?P<x>.+?)\s+lining$", re.IGNORECASE)
_ONE_PCT_RE = re.compile(r"^(?P<n>\d+)%\s+(?P<m>[A-Za-z][A-Za-z ]*)$")
_MULTI_PCT_RE = re.compile(r"^\d+%\s+[A-Za-z][A-Za-z ]*?(?:,\s*\d+%\s+[A-Za-z][A-Za-z ]*?)+$")
_PCT_PART_RE = re.compile(r"(?P<n>\d+)%\s+(?P<m>[A-Za-z][A-Za-z ]*)")

_EXACT: dict[str, list[str]] = {
    "machine wash": ["machine washable", "safe to machine wash", "can go in the washing machine"],
    "hand wash only": ["must be washed by hand", "hand washing only, no machine", "wash by hand only"],
    "imported": ["an imported item", "imported rather than domestic", "brought in from overseas"],
    "made in the usa": ["manufactured in the United States", "US-made"],
    "made in the usa or imported": [
        "either US-made or brought in from overseas",
        "manufactured in the United States, or else imported",
    ],
    "no closure closure": ["no fastening at all", "without any fastening"],
}

# Word-level swaps for the long marketing sentences (constraints of 6+ words).
# Deliberately generic English, not agent-specific vocabulary.
_WORD_SWAPS: dict[str, str] = {
    "comfortable": "comfy", "comfort": "cosiness", "lightweight": "light in weight",
    "durable": "long-lasting", "durability": "hard-wearing build",
    "design": "styling", "designed": "built", "easy": "simple",
    "soft": "gentle to the touch", "adjustable": "able to be adjusted",
    "with": "featuring", "and": "plus", "for": "suited to",
    "high": "tall", "quality": "workmanship", "perfect": "ideal",
    "great": "excellent", "classic": "timeless", "premium": "top-grade",
    "waterproof": "water-tight", "breathable": "air-permeable",
    "elastic": "stretchy", "flexible": "bendable", "slip-on": "step-in",
    "closure": "fastening", "sole": "outsole", "colour": "color",
    "women": "ladies", "men": "gentlemen", "wash": "launder",
    "approximately": "roughly", "measures": "comes in at",
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

# L3b only: the category tail. Meaning-preserving, but no longer a verbatim
# substring of the catalog's own `categories` construction.
_CATEGORY_SWAPS: dict[str, str] = {
    "shoes": "footwear", "watches": "timepieces", "wrist watches": "wristwatches",
    "clothing": "apparel", "jewelry": "jewellery", "accessories": "accessory items",
    "boots": "boot styles", "sandals": "open-toe sandals", "sneakers": "trainers",
    "handbags": "purses", "wallets": "billfolds", "earrings": "ear jewellery",
    "necklaces": "neck chains", "rings": "finger rings", "bracelets": "wrist bands",
    "shirts": "tops", "pants": "trousers", "socks": "hosiery", "hats": "headwear",
    "sunglasses": "shades", "belts": "waist belts", "bags": "carry bags",
    "novelty": "novelty-style", "athletic": "sports", "casual": "everyday",
    "fashion": "style-led", "girls": "girls'", "boys": "boys'",
    "women": "ladies", "men": "gentlemen", "baby": "infant",
}


def _swap_words(text: str, table: dict[str, str], rng: random.Random, rate: float) -> str:
    def replace(match: re.Match) -> str:
        word = match.group(0)
        target = table.get(word.lower())
        if target is None or rng.random() >= rate:
            return word
        return target.capitalize() if word[0].isupper() and target[0].islower() else target

    return _WORD_RE.sub(replace, text)


def rewrite_category(text: str, rng: random.Random) -> str:
    """L3b: reword the category tail so the exact anchor key no longer matches."""
    swapped = _swap_words(text, _CATEGORY_SWAPS, rng, 1.0)
    if swapped.lower() == text.lower():
        # No known term: fall back to a generic wrapper that still reads as the
        # same category to a human but is not the catalog's own string.
        return rng.choice([f"{text} items", f"the {text} range", f"{text}-type products"])
    return swapped


def rewrite_constraint(value: str, rng: random.Random, mutate_singles: bool) -> str:
    """One meaning-preserving rewrite. Returns `value` unchanged when the rule
    set has nothing faithful to say about this shape."""
    text = value.strip()
    lowered = text.lower()

    fixed = _EXACT.get(lowered)
    if fixed:
        return rng.choice(fixed)

    match = _COLOR_RE.match(text)
    if match:
        colour = match.group("c").lower()
        return rng.choice([
            f"{colour} in colour", f"coloured {colour}", f"in the colour {colour}",
        ])

    if _MULTI_PCT_RE.match(text):
        parts = [(m.group("n"), m.group("m").strip()) for m in _PCT_PART_RE.finditer(text)]
        rendered = [f"{material} ({number} percent)" for number, material in parts]
        return " blended with ".join(rendered)

    match = _ONE_PCT_RE.match(text)
    if match:
        number, material = match.group("n"), match.group("m").strip()
        return rng.choice([
            f"{material} ({number} percent)",
            f"made from {number} percent {material}",
            f"{number} percent {material} content",
        ])

    match = _SOLE_RE.match(text)
    if match:
        body = match.group("x")
        return rng.choice([
            f"soles made of {body}", f"{body} outsole", f"sole made from {body}",
        ])

    match = _CLOSURE_RE.match(text)
    if match:
        body = match.group("x")
        return rng.choice([
            f"{body.lower()} fastening", f"fastens with a {body.lower()}",
            f"{body.lower()}-style fastening",
        ])

    match = _LINING_RE.match(text)
    if match:
        return rng.choice([f"lined with {match.group('x')}", f"{match.group('x')} on the lining"])

    words = text.split()
    if len(words) == 1:
        if not mutate_singles:
            return text  # the paraphrase-proof floor — see module docstring
        head = words[0].lower()
        if head in _MATERIAL_WORDS:
            return rng.choice([f"made of {head}", f"{head} fabric", f"a {head} construction"])
        return rng.choice([f"{head} goods", f"described as {head}"])

    if len(words) >= 6:
        return _swap_words(text, _WORD_SWAPS, rng, 0.7)

    # Short multi-word phrase with no structural rule: light generic rewording.
    swapped = _swap_words(text, _WORD_SWAPS, rng, 1.0)
    if swapped.lower() != lowered:
        return swapped
    if len(words) == 2:
        # Hyphenation and a light carrier phrase both read naturally and both
        # break the whole-payload substring test without changing the meaning.
        return rng.choice([
            f"{words[0]}-{words[1]}", f"{words[0]} {words[1].lower()}s",
            f"offers {text.lower()}",
        ])
    # 3-5 words, no structural rule: a carrier phrase. Fully readable and
    # order-preserving, but it defeats a matcher that tests the *whole* payload
    # as one substring — which is the mechanism under test.
    return rng.choice([
        f"with {text}", f"featuring {text}", f"{text} included",
        f"it should have {text.lower()}",
    ])


class L3Paraphraser(Paraphraser):
    """L2's frame rewriting plus semantic payload rewrites."""

    def __init__(self, level: str = "L3a", seed: int = 0, mutate_singles: bool = False,
                 cache_path: str = "data/paraphrase_cache.jsonl") -> None:
        if level not in L3_LEVELS:
            raise ValueError(f"unknown L3 level: {level}")
        super().__init__(level="L2", seed=seed, cache_path=cache_path)
        self.l3_level = level
        self.mutate_singles = mutate_singles

    def _mutate_payload(self, shape: str, key: str, value: str, rng: random.Random) -> str:
        if key == "attr":
            return value
        if key == "cat":
            drift = self.l3_level in ("L3b", "catdrift")
            return rewrite_category(value, rng) if drift else value
        if key == "old":
            # The intent_override opening's `old_value` is a real constraint
            # (soft[-1]); L2 never touched it, which is part of why L2 is toothless.
            return self._rewrite(value, rng)
        if key == "payload":
            constraints = [self._rewrite(item, rng) for item in value.split("; ")]
            if len(constraints) >= 2 and rng.random() < 0.5:
                rng.shuffle(constraints)
            return rng.choice(L2_JOINERS).join(constraints)
        if key == "constraint":
            return self._rewrite(value, rng)
        return value

    def _rewrite(self, value: str, rng: random.Random) -> str:
        if self.l3_level == "catdrift":
            return value  # category drift only; payloads stay verbatim
        budget = BUDGET_CONSTRAINT_RE.match(value)
        if budget:
            return rng.choice(BUDGET_VARIANTS).format(amt=budget.group("amt"))
        if rng.random() >= MUTATION_RATE:
            return value
        return rewrite_constraint(value, rng, self.mutate_singles)

    def _render(self, shape: str, groups: dict[str, str], rng: random.Random) -> str:
        templates = list(L1_TEMPLATES[shape]) + list(L2_EXTRA_TEMPLATES.get(shape, []))
        groups = {key: self._mutate_payload(shape, key, value, rng)
                  for key, value in groups.items()}
        return rng.choice(templates).format(**groups)

    def paraphrase(self, message: str, sample_id: str, turn: int) -> str:
        if message in self.cache:
            return self.cache[message]
        for shape, pattern in self.__class__._shapes():
            match = pattern.match(message)
            if match:
                rng = random.Random(f"{self.seed}\0{sample_id}\0{turn}\0{self.l3_level}")
                return self._render(shape, match.groupdict(), rng)
        return message

    @staticmethod
    def _shapes():
        from tools.paraphraser import SHAPES
        return SHAPES
