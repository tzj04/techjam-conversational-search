import json
import tempfile
import unittest
from pathlib import Path

import submission.agent as agent_module
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


class GatingTest(unittest.TestCase):
    def test_uninformed_turn_returns_single_candidate(self):
        agent = build_agent()
        agent.reset("s", {})
        response = agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        self.assertEqual(len(response["recommendations"]), 1)

    def test_informed_turn_three_returns_full_list(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "For that, what matters is: 100% Cotton; Machine wash, cold.", 2, 10)
        response = agent.respond("s", "For that, what matters is: Premium comfort feel all day.", 3, 10)
        self.assertEqual(len(response["recommendations"]), 3)  # full mini-catalog

    def test_drained_escape_returns_full_list(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        response = agent.respond("s", "I don't have an additional preference for other.", 2, 10)
        self.assertEqual(len(response["recommendations"]), 3)

    def test_turn_cap_escape_returns_full_list(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        response = agent.respond("s", "unrecognizable paraphrased customer message here", 5, 10)
        self.assertEqual(len(response["recommendations"]), 3)

    def test_gating_flag_off_restores_full_lists(self):
        import submission.agent as module

        original = module.FEATURE_GATING
        module.FEATURE_GATING = False
        try:
            agent = build_agent()
            agent.reset("s", {})
            response = agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
            self.assertEqual(len(response["recommendations"]), 3)
        finally:
            module.FEATURE_GATING = original


class StalePenaltyTest(unittest.TestCase):
    def test_no_penalty_without_dissatisfaction(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "For that, what matters is: 100% Cotton.", 2, 10)
        state = agent._sessions["s"]
        self.assertTrue(state.stale_shown)          # items were shown…
        self.assertEqual(state.penalized, frozenset())  # …but nothing is penalized

    def test_penalty_applies_only_after_dissatisfaction(self):
        agent = build_agent()
        agent.reset("s", {})
        first = agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        shown = first["recommendations"][0]["parent_asin"]
        agent.respond(
            "s", "Those options are not quite right yet. Ask me about one specific attribute.", 2, 10
        )
        state = agent._sessions["s"]
        self.assertIn(shown, state.penalized)
        self.assertTrue(state.dissatisfied)


class RelaxationTest(unittest.TestCase):
    def test_dissatisfaction_relaxes_weakest_constraint_only(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond(
            "s", "For that, what matters is: Premium comfort feel all day; 100% Cotton.", 2, 10
        )
        agent.respond(
            "s", "Those options are not quite right yet. Ask me about one specific attribute.", 3, 10
        )
        state = agent._sessions["s"]
        by_norm = {constraint.norm: constraint for constraint in state.constraints}
        self.assertTrue(by_norm["100% cotton"].demoted)          # fewer tokens → weakest
        self.assertFalse(by_norm["premium comfort feel all day"].demoted)

    def test_sole_constraint_is_never_relaxed(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", "For that, what matters is: 100% Cotton.", 2, 10)
        agent.respond(
            "s", "Those options are not quite right yet. Ask me about one specific attribute.", 3, 10
        )
        self.assertFalse(agent._sessions["s"].constraints[0].demoted)


class DemoGuardTest(unittest.TestCase):
    def test_greeting_returns_no_recommendations(self):
        agent = build_agent()
        agent.reset("s", {})
        response = agent.respond("s", "Hi", 1, 10)
        self.assertEqual(response["recommendations"], [])
        self.assertEqual(response["ask_attribute"], "other")

    def test_real_opening_is_untouched_by_guard(self):
        agent = build_agent()
        agent.reset("s", {})
        response = agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        self.assertTrue(response["recommendations"])


class SoftRoutingTest(unittest.TestCase):
    def test_p_buy_tracks_message_shapes(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        self.assertAlmostEqual(agent._sessions["s"].p_buy, 0.15)
        agent.reset("b", {})
        agent.respond("b", "I'm looking for Women Dresses. A key requirement is: 100% Cotton.", 1, 10)
        self.assertGreaterEqual(agent._sessions["b"].p_buy, 0.95)
        agent.respond("b", "Actually, ignore my earlier preference. What I need is: Machine wash, cold.", 2, 10)
        self.assertGreaterEqual(agent._sessions["b"].p_buy, 0.9)  # override + new-constraint mass

    def test_profile_prior_only_on_zero_evidence_turns(self):
        import submission.agent as module

        original = module.FEATURE_PROFILE_PRIOR
        module.FEATURE_PROFILE_PRIOR = True  # cut from the shipped set; feature still tested
        try:
            agent = build_agent()
            agent.reset("s", {"preference_tags": ["comfort"]})
            agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
            state = agent._sessions["s"]
            ranked = state.cache_ranking
            self.assertEqual(ranked[0], "A001")  # only dress whose text carries "comfort"
            agent.respond("s", "For that, what matters is: Lightweight polyester shell.", 2, 10)
            # With evidence present the prior is off: constraint match decides.
            self.assertEqual(agent._sessions["s"].cache_ranking[0], "A003")
        finally:
            module.FEATURE_PROFILE_PRIOR = original


if __name__ == "__main__":
    unittest.main()


class BudgetMisroutingGuardTest(unittest.TestCase):
    """A money *word* inside a long features string is not a budget disclosure.
    Before the guard it was routed to the price path, its text discarded and a
    bogus filter applied from whatever digit appeared first."""

    LONG = (
        "WELL PRICED, TIMELESS STYLE - Traditional in its design, this "
        "inexpensive but very durable 100% Cotton shirt is built to last"
    )

    def test_long_priced_string_stays_lexical(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        agent.respond("s", f"For that, what matters is: {self.LONG}.", 2, 10)
        state = agent._sessions["s"]
        self.assertIsNone(state.budget_point)
        self.assertEqual(len(state.constraints), 1)
        self.assertFalse(state.constraints[0].is_budget)
        self.assertIn("cotton", state.constraints[0].norm)

    def test_real_budget_disclosures_still_route_to_price(self):
        for phrase in (
            "budget around $79.99",
            "my budget's about 79.99 dollars",
            "somewhere around $79.99 works for me",
            "I can spend around $79.99",
        ):
            with self.subTest(phrase=phrase):
                agent = build_agent()
                agent.reset("s", {})
                agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
                agent.respond("s", f"For that, what matters is: {phrase}.", 2, 10)
                self.assertEqual(agent._sessions["s"].budget_point, 79.99)


class PartialMatchTest(unittest.TestCase):
    def test_coverage_is_one_for_identical_and_zero_for_disjoint(self):
        agent = build_agent()
        agent_module.FEATURE_PARTIAL_IDF_FLOOR = False
        try:
            text = agent._token_text["A001"]
            self.assertAlmostEqual(agent._coverage("100% cotton", text), 1.0)
            self.assertEqual(agent._coverage("zzzz qqqq", text), 0.0)
            # A reworded payload keeps partial credit rather than scoring zero.
            boot = agent._token_text["A002"]
            self.assertGreater(agent._coverage("soles made of rubber", boot), 0.0)
        finally:
            agent_module.FEATURE_PARTIAL_IDF_FLOOR = True

    def test_idf_floor_scales_with_the_catalog(self):
        """The floor is 'one token at 5% document frequency', evaluated against
        whatever catalog is loaded — not a constant that assumes 50k rows."""
        agent = build_agent()
        self.assertEqual(agent._n_docs, len(MINI_CATALOG))
        expected = __import__("math").log(
            (agent._n_docs + 1.0)
            / (agent_module.PARTIAL_IDF_QUANTILE * agent._n_docs + 1.0)
        )
        self.assertAlmostEqual(agent._partial_min_idf, expected)

    def test_idf_floor_suppresses_uninformative_partial_evidence(self):
        """Matching only common words must score zero, not a high ratio."""
        agent = build_agent()
        agent._n_docs = 50000
        agent._doc_freq = {"made": 40000, "of": 45000, "unobtanium": 1}
        text = " made of "
        self.assertEqual(agent._coverage("made of", text), 0.0)
        self.assertGreater(agent._coverage("unobtanium", " unobtanium "), 0.0)

    def test_reworded_payload_still_ranks_the_right_product(self):
        """'Rubber sole' -> 'soles made of Rubber' is a total loss under the
        whole-string substring test; graded coverage keeps the signal."""
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Men Boots.", 1, 10)
        response = agent.respond("s", "What matters to me is: soles made of Rubber.", 2, 10)
        agent._sessions["s"].drained = 1  # force full-width output
        response = agent.respond("s", "I don't have an additional preference for other.", 3, 10)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "A002")

    def test_full_substring_match_outscores_any_partial(self):
        agent = build_agent()
        weight = 3.0
        best_partial = weight * agent_module.PARTIAL_SCALE * 1.0 * 1.0
        self.assertLess(best_partial, weight)


class FuzzyAnchorTest(unittest.TestCase):
    def test_exact_category_takes_the_full_bonus(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        state = agent._sessions["s"]
        self.assertEqual(state.anchor_bonus, agent_module.ANCHOR_BONUS)
        self.assertIn("A001", state.anchor_set)

    def test_reworded_category_feeds_recall_but_never_the_score(self):
        """A wrong fuzzy pick must cost nothing but recall: the closest keys
        enter the candidate pool, and no anchor bonus is awarded."""
        agent = build_agent()
        agent.reset("s", {})
        # "Frocks" is not a catalog category, so neither the suffix nor the
        # exact infix scan can recover it and the fuzzy path is what runs.
        agent_module.FEATURE_FUZZY_ANCHOR = True
        self.addCleanup(setattr, agent_module, "FEATURE_FUZZY_ANCHOR", False)
        agent.respond("s", "I'm looking for Women Frocks, but I'm still exploring.", 1, 10)
        state = agent._sessions["s"]
        self.assertIsNotNone(state.anchor)
        self.assertIn("A001", state.anchor)
        self.assertEqual(state.anchor_set, frozenset())
        self.assertEqual(state.anchor_bonus, 0.0)


class RegimeEscapeTest(unittest.TestCase):
    def test_gate_opens_when_nothing_matches(self):
        """Constraints held but the leader matches none of them: the exact
        matcher has failed, so deferring to GATE_CAP_TURN only burns turns."""
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        response = agent.respond(
            "s", "What matters to me is: qqqqzzz wwwwvvv; xxxxyyy uuuuttt.", 2, 10
        )
        self.assertGreater(len(response["recommendations"]), 1)

    def test_gate_still_defers_when_matching_works(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses, but I'm still exploring.", 1, 10)
        response = agent.respond("s", "What matters to me is: 100% Cotton; Machine wash, cold.", 2, 10)
        self.assertEqual(len(response["recommendations"]), agent_module.GATE_DEPTH)


class FieldWeightTest(unittest.TestCase):
    def test_field_bonus_respects_the_demotion_asymmetry(self):
        """Demoted override evidence is kept at 0.1x weight. A flat field bonus
        would hand it back full strength; a proportional one scales with it."""
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses. Machine wash, cold", 1, 10)
        state = agent._sessions["s"]
        live = [c for c in state.constraints if not c.is_budget][0]
        full = live.weight * agent_module.FIELD_WEIGHT_RATIO
        agent.respond("s", "Actually, ignore my earlier preference. What I need is: 100% Cotton.", 2, 10)
        demoted = [c for c in state.constraints if c.norm == live.norm][0]
        self.assertTrue(demoted.demoted)
        self.assertAlmostEqual(demoted.weight * agent_module.FIELD_WEIGHT_RATIO, full * 0.1)


class InfixAnchorTest(unittest.TestCase):
    def test_category_mid_sentence_still_anchors(self):
        """The suffix scan needs the category to end a clause; a frame that
        appends text with no delimiter must not lose the anchor."""
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm shopping for Women Dresses but haven't settled on anything.", 1, 10)
        state = agent._sessions["s"]
        self.assertIn("A001", state.anchor_set)
        self.assertEqual(state.anchor_bonus, agent_module.ANCHOR_BONUS)

    def test_infix_prefers_the_longest_exact_key(self):
        agent = build_agent()
        self.assertEqual(agent._anchor_infix("I want Women Dresses today"), "women dresses")

    def test_infix_never_matches_a_lone_short_token(self):
        agent = build_agent()
        self.assertIsNone(agent._anchor_infix("I am on it"))


class MultiQueryRetrievalTest(unittest.TestCase):
    def test_each_constraint_contributes_candidates_independently(self):
        """A single OR-of-everything query lets BM25 favour documents matching
        many common terms; a rare constraint can then contribute nothing."""
        agent = build_agent()
        agent_module.FEATURE_MULTI_QUERY = True
        try:
            agent.reset("s", {})
            agent.respond("s", "I'm looking for Men Boots.", 1, 10)
            state = agent._sessions["s"]
            state.anchor = None
            state.anchor_set = frozenset()
            agent._add_constraint(state, "Waterproof membrane")
            pool = agent._build_pool(state, 10)
            self.assertIn("A002", pool)
        finally:
            agent_module.FEATURE_MULTI_QUERY = False
