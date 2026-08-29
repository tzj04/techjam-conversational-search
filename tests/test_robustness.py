"""Deployment- and contract-level robustness of the submission agent.

Every case here is a way the agent scored **zero** on an organizer harness
without any of the public-set numbers changing: the catalog resolved to a path
that did not exist, one malformed row, no FTS5 in the host SQLite build, a
worker thread, a field typed as a string. None of them is reachable from
`tools/run_eval.py`, which is exactly why they were missed.
"""
import json
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import submission.agent as agent_module
from submission.agent import Agent, _fold, _number, _pivot_window, _terms, normalize_text

from tests.test_submission_agent import MINI_CATALOG, build_agent


def write_catalog(rows, extra_lines=(), name="catalog.jsonl") -> Path:
    path = Path(tempfile.mkdtemp()) / name
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        for line in extra_lines:
            handle.write(line + "\n")
    return path


class CatalogResolutionTest(unittest.TestCase):
    """`Agent()` defaults to a *relative* path. A harness that constructs the
    agent from any other working directory used to index nothing and return an
    empty list for all 800 sessions, with no error anywhere."""

    def test_env_var_resolves_when_the_relative_default_misses(self):
        path = write_catalog(MINI_CATALOG)
        os.environ["TECHJAM_CATALOG"] = str(path)
        self.addCleanup(os.environ.pop, "TECHJAM_CATALOG", None)
        agent = Agent("data/does-not-exist.jsonl")
        self.assertEqual(agent.catalog_path, path)
        self.assertTrue(agent.catalog_loaded)
        self.assertEqual(len(agent._products), len(MINI_CATALOG))

    def test_module_relative_fallback_is_independent_of_the_cwd(self):
        """The last resort is the repository layout around this module, which
        does not move when the process starts somewhere else."""
        root = Path(agent_module.__file__).resolve().parent.parent
        expected = root / "data" / "catalog.jsonl"
        if expected.exists():  # a real catalog is present; nothing to stage
            self.assertEqual(Agent._resolve_catalog("data/gone.jsonl"), expected)
            return
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_text(json.dumps(MINI_CATALOG[0]) + "\n", encoding="utf-8")
        self.addCleanup(expected.unlink)
        self.assertEqual(Agent._resolve_catalog("data/gone.jsonl"), expected)

    def test_an_unresolvable_path_is_returned_unchanged(self):
        """So the warning names the path the caller actually asked for."""
        self.assertEqual(
            Agent._resolve_catalog("/nonexistent/nowhere.jsonl"),
            Path("/nonexistent/nowhere.jsonl"),
        )

    def test_an_explicit_existing_path_always_wins(self):
        path = write_catalog(MINI_CATALOG)
        other = write_catalog(MINI_CATALOG[:1])
        os.environ["TECHJAM_CATALOG"] = str(other)
        self.addCleanup(os.environ.pop, "TECHJAM_CATALOG", None)
        self.assertEqual(Agent(path).catalog_path, path)

    def test_a_missing_catalog_is_reported_on_stderr(self):
        import io
        import contextlib

        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            agent = Agent("/nonexistent/catalog.jsonl")
        self.assertFalse(agent.catalog_loaded)
        self.assertIn("catalog not found", buffer.getvalue())


class MalformedCatalogTest(unittest.TestCase):
    """One bad row must not cost the run. These all raised out of `__init__`."""

    def test_unparseable_line_is_skipped(self):
        agent = Agent(write_catalog(MINI_CATALOG, extra_lines=["{not json"]))
        self.assertEqual(len(agent._products), len(MINI_CATALOG))
        self.assertEqual(agent.skipped_rows, 1)

    def test_row_without_parent_asin_is_skipped(self):
        agent = Agent(write_catalog(MINI_CATALOG + [{"title": "orphan"}]))
        self.assertEqual(len(agent._products), len(MINI_CATALOG))
        self.assertEqual(agent.skipped_rows, 1)

    def test_non_object_row_is_skipped(self):
        agent = Agent(write_catalog(MINI_CATALOG, extra_lines=["[1, 2, 3]", '"text"']))
        self.assertEqual(len(agent._products), len(MINI_CATALOG))
        self.assertEqual(agent.skipped_rows, 2)

    def test_decorated_numeric_fields_do_not_break_the_build(self):
        rows = [
            {**MINI_CATALOG[0], "parent_asin": "N1", "average_rating": "4.5 out of 5 stars"},
            {**MINI_CATALOG[0], "parent_asin": "N2", "rating_number": "1,234"},
            {**MINI_CATALOG[0], "parent_asin": "N3", "average_rating": None},
        ]
        agent = Agent(write_catalog(rows))
        self.assertEqual(len(agent._products), 3)
        self.assertEqual(agent.skipped_rows, 0)
        # "4.5 out of 5 stars" must sort as 4.5, ahead of the unrated row.
        self.assertLess(
            agent._fallback_order.index("N1"), agent._fallback_order.index("N3")
        )


class NumberCoercionTest(unittest.TestCase):
    def test_shapes_amazon_actually_ships(self):
        self.assertEqual(_number(14.99), 14.99)
        self.assertEqual(_number(3), 3.0)
        self.assertEqual(_number("14.99"), 14.99)
        self.assertEqual(_number("$14.99"), 14.99)
        self.assertEqual(_number("1,234"), 1234.0)
        self.assertEqual(_number("4.5 out of 5 stars"), 4.5)
        self.assertIsNone(_number(None))
        self.assertIsNone(_number(""))
        self.assertIsNone(_number("unrated"))
        self.assertIsNone(_number(True), "a bool is not a rating")

    def test_string_price_earns_the_exact_budget_bonus(self):
        """The budget path used to read "$14.99" as *no price* and penalise the
        one product it was meant to match exactly."""
        agent = build_agent()
        agent.reset("s", {})
        state = agent._sessions["s"]
        state.budget_point = 14.99
        self.assertEqual(
            agent._budget_score(state, {"price": "$14.99"}),
            agent_module.BUDGET_EXACT_BONUS,
        )
        self.assertEqual(
            agent._budget_score(state, {"price": None}),
            -agent_module.BUDGET_NO_PRICE_PENALTY,
        )


class FtsUnavailableTest(unittest.TestCase):
    """CPython default builds ship FTS5, but a build without it turned the
    whole agent into an exception at construction time."""

    def test_construction_survives_and_the_anchor_still_ranks(self):
        real_connect = sqlite3.connect

        class Cursor:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def execute(self, sql, *args):
                if "fts5" in sql.lower():
                    raise sqlite3.OperationalError("no such module: fts5")
                return self._inner.execute(sql, *args)

        class Connection:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def cursor(self):
                return Cursor(self._inner.cursor())

        agent_module.sqlite3.connect = lambda *a, **k: Connection(real_connect(*a, **k))
        self.addCleanup(setattr, agent_module.sqlite3, "connect", real_connect)
        import io
        import contextlib

        with contextlib.redirect_stderr(io.StringIO()):
            agent = Agent(write_catalog(MINI_CATALOG))
        self.assertFalse(agent.fts_enabled)
        self.assertEqual(len(agent._products), len(MINI_CATALOG))
        self.assertEqual(agent._fts_search('"cotton"', 10), [])
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses. A key requirement is: 100% Cotton.", 1, 10)
        agent._sessions["s"].drained = 1
        response = agent.respond("s", "I don't have an additional preference for other.", 2, 10)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "A001")


class ThreadSafetyTest(unittest.TestCase):
    def test_respond_works_from_a_worker_thread(self):
        """sqlite3 connections are thread-affine by default. In a threaded
        harness every query raised ProgrammingError, `respond` swallowed it, and
        the run scored zero without a single visible error."""
        agent = build_agent()
        agent.reset("s", {})
        captured = {}

        def work():
            captured["response"] = agent.respond(
                "s", "I'm looking for Women Dresses. A key requirement is: 100% Cotton.", 1, 10
            )

        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        self.assertEqual(
            captured["response"]["recommendations"][0]["parent_asin"], "A001"
        )

    def test_concurrent_sessions_do_not_error(self):
        agent = build_agent()
        errors = []

        def work(index):
            session = f"s{index}"
            try:
                agent.reset(session, {})
                for turn in range(1, 4):
                    agent._respond(
                        session,
                        "I'm looking for Women Dresses. A key requirement is: 100% Cotton.",
                        turn,
                        10,
                    )
            except Exception as exc:  # pragma: no cover - the assertion reports it
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])


class FallbackResponseTest(unittest.TestCase):
    """An internal fault used to answer `recommendations: []` — a guaranteed
    zero for the turn. The evaluator applies no penalty for wrong picks, so any
    list is weakly better than none."""

    def test_fault_falls_back_to_the_last_good_ranking(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses. A key requirement is: 100% Cotton.", 1, 10)
        ranking = list(agent._sessions["s"].cache_ranking)
        self.assertTrue(ranking)

        def boom(*args, **kwargs):
            raise RuntimeError("induced")

        agent._respond = boom
        response = agent.respond("s", "For that, what matters is: Imported.", 8, 10)
        self.assertTrue(response["recommendations"])
        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ranking[:10],
        )

    def test_fault_with_no_session_falls_back_to_popularity(self):
        agent = build_agent()

        def boom(*args, **kwargs):
            raise RuntimeError("induced")

        agent._respond = boom
        response = agent.respond("unknown", "hello", 9, 10)
        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            agent._fallback_order[:10],
        )

    def test_the_fallback_respects_gating(self):
        """A fault must not lock in a bad rank the normal path would defer."""
        agent = build_agent()

        def boom(*args, **kwargs):
            raise RuntimeError("induced")

        agent._respond = boom
        early = agent.respond("unknown", "hello", 1, 10)
        late = agent.respond("unknown", "hello", agent_module.GATE_CAP_TURN, 10)
        self.assertEqual(len(early["recommendations"]), agent_module.GATE_DEPTH)
        self.assertGreater(len(late["recommendations"]), agent_module.GATE_DEPTH)

    def test_the_response_schema_survives_the_fault(self):
        agent = build_agent()

        def boom(*args, **kwargs):
            raise RuntimeError("induced")

        agent._respond = boom
        response = agent.respond("unknown", "hello", 1, 10)
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], agent_module.ALLOWED_ATTRIBUTES)
        self.assertIsInstance(response["usage"]["prompt_tokens"], int)


class ArgumentCoercionTest(unittest.TestCase):
    def test_non_integer_top_k_and_turn_are_coerced(self):
        agent = build_agent()
        for top_k, turn in [("10", 1), (None, 1), (10.0, "1"), (10, None)]:
            agent.reset("s", {})
            response = agent.respond(
                "s", "I'm looking for Women Dresses. A key requirement is: 100% Cotton.",
                turn, top_k,
            )
            self.assertTrue(response["recommendations"], (top_k, turn))

    def test_absurd_top_k_is_clamped(self):
        agent = build_agent()
        agent.reset("s", {})
        agent._sessions["s"].drained = 1
        response = agent.respond("s", "I'm looking for Women Dresses.", 9, 10 ** 9)
        self.assertLessEqual(len(response["recommendations"]), len(MINI_CATALOG))


class UnicodeFoldTest(unittest.TestCase):
    """The normal form keeps `[a-z0-9]` only; the FTS5 index is tokenized
    `remove_diacritics 2`. Without folding, the two disagree on every accented
    word and the query side emits fragments that cannot match anything."""

    def test_fold_is_the_identity_on_ascii(self):
        for text in ["100% Cotton", "Rubber sole", "budget around $79.99", ""]:
            self.assertEqual(_fold(text), text)

    def test_accented_words_survive_as_whole_tokens(self):
        self.assertEqual(normalize_text("Café"), "cafe")
        self.assertEqual(_terms("Damenmütze"), ["damenmutze"])
        self.assertEqual(_terms("naïve Bébé"), ["naive", "bebe"])

    def test_the_query_side_agrees_with_the_fts_tokenizer(self):
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE VIRTUAL TABLE t USING fts5(body, tokenize='unicode61 remove_diacritics 2')"
        )
        connection.execute("INSERT INTO t VALUES (?)", ("Café Damenmütze naïve",))
        for token in _terms("Café Damenmütze naïve"):
            rows = connection.execute(
                "SELECT rowid FROM t WHERE t MATCH ?", (f'"{token}"',)
            ).fetchall()
            self.assertTrue(rows, f"index has no token matching {token!r}")

    def test_an_accented_constraint_retrieves_its_product(self):
        rows = [{
            "parent_asin": "T001", "title": "Damenmütze Beanie",
            "categories": ["Clothing, Shoes & Jewelry", "Women", "Hats"],
            "features": ["Damenmütze"], "details": {}, "description": [],
            "store": "S", "price": 20.0, "average_rating": 3.0, "rating_number": 1,
        }]
        # Enough popular decoys to fill the fallback pad, so the target enters
        # the pool through FTS or not at all.
        rows += [{
            "parent_asin": f"D{index:03d}", "title": f"Plain Beanie {index}",
            "categories": ["Clothing, Shoes & Jewelry", "Men", "Caps"],
            "features": ["Acrylic"], "details": {}, "description": [],
            "store": "T", "price": 9.0, "average_rating": 5.0, "rating_number": 9999,
        } for index in range(400)]
        agent = Agent(write_catalog(rows))
        agent.reset("s", {})
        response = agent.respond("s", "For that, what matters is: Damenmütze.", 5, 10)
        self.assertIn(
            "T001", [item["parent_asin"] for item in response["recommendations"]]
        )


class OverridePivotTest(unittest.TestCase):
    def test_window_stops_at_the_payload_separator(self):
        self.assertEqual(
            _pivot_window("for that, what matters is: forget the rest", 60),
            "for that, what matters is",
        )

    def test_window_is_bounded_by_length(self):
        self.assertEqual(len(_pivot_window("x" * 200, 60)), 60)

    def test_a_late_pivot_mid_conversation_is_still_an_override(self):
        """Every simulator frame puts the pivot inside 20 characters, but a
        private-set lead-in need not."""
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses. A key requirement is: 100% Cotton.", 1, 10)
        agent.respond(
            "s",
            "Let me change direction here — actually, what I need is: Full-grain leather.",
            2, 10,
        )
        state = agent._sessions["s"]
        cotton = [c for c in state.constraints if c.norm == "100% cotton"]
        self.assertTrue(cotton and cotton[0].demoted, "prior evidence must be demoted")
        self.assertIn("full-grain leather", [c.verbatim.lower() for c in state.constraints])

    def test_a_payload_saying_instead_is_not_an_override(self):
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses. A key requirement is: 100% Cotton.", 1, 10)
        agent.respond("s", "For that, what matters is: wear it instead of a coat.", 2, 10)
        state = agent._sessions["s"]
        cotton = [c for c in state.constraints if c.norm == "100% cotton"]
        self.assertTrue(cotton and not cotton[0].demoted)


class OverrideBeforeOpeningTest(unittest.TestCase):
    """When the opening's category was reworded no anchor is held, so the
    override message reached `_try_opening` first. Its infix scan then read a
    category word out of the override's *payload* and installed a wrong anchor
    at the full bonus — and the override itself was never applied."""

    CATALOG = [
        {"parent_asin": f"F{index:03d}", "title": f"Frock {index}",
         "categories": ["Clothing, Shoes & Jewelry", "Women", "Sundresses"],
         "features": ["100% Cotton"], "details": {}, "description": [],
         "store": "S", "price": 20.0, "average_rating": 4.0, "rating_number": 100 + index}
        for index in range(3)
    ] + [
        {"parent_asin": f"S{index:03d}", "title": f"Sports Bra {index}",
         "categories": ["Clothing, Shoes & Jewelry", "Women", "Sports Bras"],
         "features": ["Moisture wicking"], "details": {}, "description": [],
         "store": "T", "price": 30.0, "average_rating": 4.9, "rating_number": 5000 + index}
        for index in range(3)
    ]

    def test_the_override_is_applied_and_no_anchor_is_stolen(self):
        agent = Agent(write_catalog(self.CATALOG))
        agent.reset("s", {})
        state = agent._sessions["s"]
        agent._extract(state, "I'm hunting for a womens summer frock. A key requirement is: 100% Cotton.")
        self.assertIsNone(state.anchor, "the reworded category must not resolve")

        agent._extract(
            state,
            "Actually, ignore my earlier preference. "
            "What I need is: great under Women Sports Bras tops.",
        )
        self.assertFalse(state.anchor, "no anchor may be taken from an override payload")
        self.assertEqual(state.anchor_bonus, 0.0)
        cotton = [c for c in state.constraints if c.norm == "100% cotton"]
        self.assertTrue(cotton and cotton[0].demoted, "the override must demote prior evidence")

    def test_a_first_message_is_never_read_as_an_override(self):
        agent = Agent(write_catalog(self.CATALOG))
        agent.reset("s", {})
        state = agent._sessions["s"]
        agent._extract(state, "I'm looking for Women Sundresses, but I'm still exploring.")
        self.assertTrue(state.anchor)
        self.assertEqual(state.anchor_bonus, agent_module.ANCHOR_BONUS)


class InfixAnchorScopeTest(unittest.TestCase):
    def test_the_infix_scan_never_reads_the_payload(self):
        agent = Agent(write_catalog(OverrideBeforeOpeningTest.CATALOG))
        agent.reset("s", {})
        state = agent._sessions["s"]
        # Category reworded in the lead-in; a real category name sits in the
        # payload. Anchoring on the payload would bury the target.
        agent._extract(
            state,
            "I'm hunting for a womens summer frock. "
            "A key requirement is: pairs with Women Sports Bras.",
        )
        self.assertFalse(state.anchor)

    def test_a_lead_in_category_is_still_recovered(self):
        agent = Agent(write_catalog(OverrideBeforeOpeningTest.CATALOG))
        agent.reset("s", {})
        state = agent._sessions["s"]
        # No delimiter after the category: the suffix scan fails, infix recovers.
        agent._extract(state, "I'm shopping for Women Sundresses but haven't settled on anything")
        self.assertTrue(state.anchor)
        self.assertEqual(state.anchor_bonus, agent_module.ANCHOR_BONUS)
        self.assertTrue(all(asin.startswith("F") for asin in state.anchor))


class OverrideValueSalvageTest(unittest.TestCase):
    def test_an_unrecognized_value_frame_keeps_its_words(self):
        """Demoting everything and then discarding the new preference leaves
        the session with no active evidence at all."""
        agent = build_agent()
        agent.reset("s", {})
        agent.respond("s", "I'm looking for Women Dresses. A key requirement is: 100% Cotton.", 1, 10)
        agent.respond("s", "Actually, scratch that — full-grain leather please", 2, 10)
        state = agent._sessions["s"]
        self.assertTrue(
            {"leather", "grain"} & set(state.loose_terms),
            f"the new preference was discarded: {state.loose_terms}",
        )


if __name__ == "__main__":
    unittest.main()


class SyntheticHarnessTest(unittest.TestCase):
    """`tools/synthetic_eval.py` is the only end-to-end instrument that runs
    without the 50k download, so it has to keep working."""

    def test_a_small_run_is_deterministic_and_scores(self):
        from tools import synthetic_eval

        first = synthetic_eval.run_one("submission.agent", "clean", 0, 200, 12)
        second = synthetic_eval.run_one("submission.agent", "clean", 0, 200, 12)
        self.assertEqual(synthetic_eval.digest(first), synthetic_eval.digest(second))
        self.assertEqual(first["sample_count"], 12)
        self.assertGreater(first["hit_rate_at_10"], 0.0)

    def test_category_drift_rewrites_the_category_and_nothing_else(self):
        from tools.l3_paraphraser import L3Paraphraser

        paraphraser = L3Paraphraser(level="catdrift", seed=0)
        rewritten = paraphraser.paraphrase(
            "I'm looking for Women Dresses. A key requirement is: 100% Cotton.",
            "synth_0000", 1,
        )
        self.assertIn("100% Cotton", rewritten)
        self.assertNotIn("Women Dresses", rewritten)
