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


class QuestionPolicyTest(unittest.TestCase):
    def test_null_only_after_two_no_additional_replies(self):
        agent = build_agent()
        agent.reset("s", {})
        first = agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        self.assertEqual(first["ask_attribute"], "other")
        second = agent.respond("s", "I don't have an additional preference for other.", 2, 10)
        self.assertEqual(second["ask_attribute"], "other")
        third = agent.respond("s", "I don't have an additional preference for other.", 3, 10)
        self.assertIsNone(third["ask_attribute"])

    def test_boundary_no_preference_for_other_does_not_stop_asks(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        response = agent.respond(
            "s", "I don't have a preference for other; please use your judgment.", 2, 10
        )
        self.assertEqual(response["ask_attribute"], "other")
        self.assertIn("other", agent._sessions["s"].exhausted)
        self.assertEqual(agent._sessions["s"].drained, 0)

    def test_drained_resets_on_contentful_reply(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "I don't have an additional preference for other.", 2, 10)
        self.assertEqual(agent._sessions["s"].drained, 1)
        agent.respond("s", "For that, what matters is: 100% Cotton.", 3, 10)
        self.assertEqual(agent._sessions["s"].drained, 0)

    def test_dissatisfaction_resumes_asking(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "I don't have an additional preference for other.", 2, 10)
        agent.respond("s", "I don't have an additional preference for other.", 3, 10)
        response = agent.respond(
            "s", "Those options are not quite right yet. Ask me about one specific attribute.", 4, 10
        )
        self.assertEqual(response["ask_attribute"], "other")


class OverrideRecoveryTest(unittest.TestCase):
    def _session_with_override(self) -> Agent:
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "For that, what matters is: 100% Cotton.", 2, 10)
        return agent

    def test_override_demotes_not_deletes(self):
        agent = self._session_with_override()
        old_weight = agent._sessions["s"].constraints[0].weight
        agent.respond(
            "s", "Actually, ignore my earlier preference. What I need is: Lightweight polyester shell.", 3, 10
        )
        state = agent._sessions["s"]
        old = state.constraints[0]
        self.assertTrue(old.demoted)
        self.assertAlmostEqual(old.weight, old_weight * 0.1)
        self.assertFalse(state.constraints[1].demoted)
        # Matched demoted evidence still contributes: A001 (matches old only)
        # outscores A002 (matches neither) on the pure constraint score.
        self.assertGreater(state.memo["A001"][1], state.memo["A002"][1])

    def test_consistency_gate_unmatched_demoted_costs_nothing(self):
        agent = self._session_with_override()
        agent.respond(
            "s", "Actually, ignore my earlier preference. What I need is: Lightweight polyester shell.", 3, 10
        )
        state = agent._sessions["s"]
        # A002 matches neither constraint: only the ACTIVE one penalizes.
        from submission.agent import UNMATCHED_PENALTY

        self.assertAlmostEqual(state.memo["A002"][1], -UNMATCHED_PENALTY)

    def test_memo_invalidated_on_override(self):
        agent = self._session_with_override()
        before = agent._sessions["s"].memo["A001"][1]
        self.assertGreater(before, 0)  # full-weight match on "100% cotton"
        agent.respond(
            "s", "Actually, ignore my earlier preference. What I need is: Lightweight polyester shell.", 3, 10
        )
        after = agent._sessions["s"].memo["A001"][1]
        self.assertLess(after, before)  # rescored under demotion, not stale

    def test_override_ranks_new_value_first_and_clears_stale(self):
        agent = self._session_with_override()
        self.assertTrue(agent._sessions["s"].stale_shown)
        response = agent.respond(
            "s", "Actually, ignore my earlier preference. What I need is: Lightweight polyester shell.", 3, 10
        )
        self.assertEqual(response["recommendations"][0]["parent_asin"], "A003")
        # stale_shown was cleared on override, then repopulated by this turn.
        self.assertEqual(
            agent._sessions["s"].stale_shown,
            {item["parent_asin"] for item in response["recommendations"]},
        )

    def test_override_drops_budget_unless_reasserted(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "For that, what matters is: budget around $79.99.", 2, 10)
        self.assertEqual(agent._sessions["s"].budget_point, 79.99)
        agent.respond(
            "s", "Actually, ignore my earlier preference. What I need is: Machine wash, cold.", 3, 10
        )
        self.assertIsNone(agent._sessions["s"].budget_point)


class RobustExtractionTest(unittest.TestCase):
    def test_suffix_anchor_scan_under_reworded_opening(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "Help me track down Women Dresses — just browsing for now.", 1, 10)
        self.assertEqual(set(agent._sessions["s"].anchor_set), {"A001", "A003"})

    def test_reworded_opening_keeps_trailing_constraint(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm shopping for Women Dresses. Lightweight polyester shell", 1, 10)
        state = agent._sessions["s"]
        self.assertEqual(set(state.anchor_set), {"A001", "A003"})
        self.assertEqual(len(state.constraints), 1)
        self.assertIn(state.constraints[0].norm, agent._norm_text["A003"])

    def test_reworded_reply_frame_still_captured(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "Here's what I care about: Machine wash, cold.", 2, 10)
        state = agent._sessions["s"]
        self.assertEqual(len(state.constraints), 1)
        self.assertIn(state.constraints[0].norm, agent._norm_text["A001"])

    def test_case_flipped_constraint_still_matches(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "For that, what matters is: 100% cotton; machine wash, cold.", 2, 10)
        state = agent._sessions["s"]
        self.assertEqual(len(state.constraints), 2)
        for constraint in state.constraints:
            self.assertIn(constraint.norm, agent._norm_text["A001"])

    def test_secondary_joiner_splits_payload(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "For that, what matters is: 100% Cotton and also Machine wash, cold.", 2, 10)
        state = agent._sessions["s"]
        self.assertEqual(len(state.constraints), 2)
        for constraint in state.constraints:
            self.assertIn(constraint.norm, agent._norm_text["A001"])

    def test_reworded_override_still_demotes(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "For that, what matters is: 100% Cotton.", 2, 10)
        agent.respond(
            "s", "On second thought, forget what I said earlier. What I need is: Lightweight polyester shell.", 3, 10
        )
        state = agent._sessions["s"]
        self.assertTrue(state.constraints[0].demoted)
        self.assertFalse(state.constraints[1].demoted)


if __name__ == "__main__":
    unittest.main()
