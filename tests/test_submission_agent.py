import json
import tempfile
import unittest
from pathlib import Path

from submission.agent import (
    ALLOWED_ATTRIBUTES,
    Agent,
    extract_budget_point,
    normalize_text,
)


MINI_CATALOG = [
    {
        "parent_asin": "A001",
        "title": "Classic Cotton Summer Dress",
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Dresses"],
        "features": ["100% Cotton", "Machine wash, cold", "Premium comfort feel all day"],
        "details": {"Fabric type": "100% Cotton", "Care instructions": "Machine wash"},
        "description": ["A breezy dress."],
        "store": "SunThreads",
        "price": 79.99,
        "average_rating": 4.5,
        "rating_number": 320,
    },
    {
        "parent_asin": "A002",
        "title": "Leather Hiking Boot",
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Boots"],
        "features": ["Full-grain leather", "Waterproof membrane"],
        "details": {"Sole material": "Rubber"},
        "description": [],
        "store": "TrailCo",
        "price": 129.5,
        "average_rating": 4.2,
        "rating_number": 87,
    },
    {
        "parent_asin": "A003",
        "title": "Polyester Rain Jacket",
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Dresses"],
        "features": ["Lightweight polyester shell"],
        "details": {},
        "description": [],
        "store": "DryFit",
        "price": 45.0,
        "average_rating": 3.9,
        "rating_number": 12,
    },
]


def build_agent() -> Agent:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    with handle:
        for product in MINI_CATALOG:
            handle.write(json.dumps(product) + "\n")
    return Agent(Path(handle.name))


class NormalFormTest(unittest.TestCase):
    def test_colon_flatten_matches_space_flatten(self):
        self.assertEqual(normalize_text("Fabric type: Value"), normalize_text("Fabric type Value"))

    def test_numbers_keep_symbols(self):
        self.assertEqual(normalize_text("Budget around $79.99!"), "budget around $79.99")
        self.assertIn("100% cotton", normalize_text("Soft, 100% Cotton — machine wash."))

    def test_truncated_tail_still_substring_matches(self):
        text = normalize_text("Premium comfort feel all day")
        truncated = normalize_text("Premium comfort fee")  # mid-word tail truncation
        self.assertIn(truncated, text)


class BudgetExtractionTest(unittest.TestCase):
    def test_paraphrased_budget_phrasings(self):
        self.assertEqual(extract_budget_point("my budget's about $80"), 80.0)
        self.assertEqual(extract_budget_point("around 79.99 dollars"), 79.99)

    def test_numberless_budget_gives_no_filter(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "For that, what matters is: budget around fifty dollars.", 2, 10)
        state = agent._sessions["s"]
        self.assertIsNone(state.budget_point)
        self.assertEqual(len(state.constraints), 0)


class ConstraintCaptureTest(unittest.TestCase):
    def test_two_constraint_reply_matches_both(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "For that, what matters is: 100% Cotton; Machine wash, cold.", 2, 10)
        state = agent._sessions["s"]
        self.assertEqual(len(state.constraints), 2)
        target_text = agent._norm_text["A001"]
        for constraint in state.constraints:
            self.assertIn(constraint.norm, target_text)

    def test_budget_constraint_never_substring_matched(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "For that, what matters is: budget around $79.99.", 2, 10)
        state = agent._sessions["s"]
        self.assertEqual(state.budget_point, 79.99)
        self.assertEqual(len(state.constraints), 1)
        self.assertTrue(state.constraints[0].is_budget)
        self.assertEqual(state.constraints[0].norm, "")

    def test_no_preference_adds_no_constraint(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        before = len(agent._sessions["s"].constraints)
        agent.respond("s", "I don't have a preference for color; please use your judgment.", 2, 10)
        agent.respond("s", "I don't have an additional preference for other.", 3, 10)
        self.assertEqual(len(agent._sessions["s"].constraints), before)

    def test_anchor_locks_to_category(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses. A key requirement is: 100% Cotton.", 1, 10)
        state = agent._sessions["s"]
        self.assertEqual(set(state.anchor_set), {"A001", "A003"})
        self.assertEqual(len(state.constraints), 1)
        self.assertEqual(state.constraints[0].norm, "100% cotton")


class RespondSchemaTest(unittest.TestCase):
    def test_schema_valid_across_ten_turns(self):
        agent = build_agent()
        agent.reset("s", {"preference_tags": ["comfort"]})
        messages = [
            "I'm looking for Women Dresses. Lightweight polyester shell",
            "Actually, ignore my earlier preference. What I need is: 100% Cotton.",
            "For that, what matters is: Machine wash, cold; budget around $79.99.",
            "I don't have an additional preference for other.",
            "Those options are not quite right yet. Ask me about one specific attribute.",
            "Hi",
        ]
        for turn in range(1, 11):
            message = messages[(turn - 1) % len(messages)]
            response = agent.respond("s", message, turn, 10)
            self.assertIsInstance(response, dict)
            self.assertIsInstance(response["message"], str)
            self.assertIn(response["ask_attribute"], ALLOWED_ATTRIBUTES | {None})
            self.assertIsInstance(response["recommendations"], list)
            self.assertLessEqual(len(response["recommendations"]), 10)
            for item in response["recommendations"]:
                self.assertIn(item["parent_asin"], {"A001", "A002", "A003"})
            self.assertGreaterEqual(response["usage"]["prompt_tokens"], 0)
            self.assertGreaterEqual(response["usage"]["completion_tokens"], 0)

    def test_ranking_prefers_full_constraint_match(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        response = agent.respond(
            "s", "For that, what matters is: 100% Cotton; budget around $79.99.", 2, 10
        )
        self.assertEqual(response["recommendations"][0]["parent_asin"], "A001")


if __name__ == "__main__":
    unittest.main()
