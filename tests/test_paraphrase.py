import json
import tempfile
import unittest
from pathlib import Path

from tools.paraphraser import Paraphraser


REPLY = "For that, what matters is: 100% Cotton; Machine Wash Cold."
OPENING = "I'm looking for Women Dresses, but I'm still exploring."


class ParaphraserTests(unittest.TestCase):
    def test_deterministic_for_fixed_seed(self):
        first = Paraphraser(level="L2", seed=7, cache_path="missing.jsonl")
        second = Paraphraser(level="L2", seed=7, cache_path="missing.jsonl")
        for turn in range(1, 11):
            self.assertEqual(
                first.paraphrase(REPLY, "sample_1", turn),
                second.paraphrase(REPLY, "sample_1", turn),
            )

    def test_l1_preserves_constraint_payload_verbatim(self):
        paraphraser = Paraphraser(level="L1", seed=0, cache_path="missing.jsonl")
        for turn in range(1, 11):
            output = paraphraser.paraphrase(REPLY, "sample_1", turn)
            self.assertIn("100% Cotton; Machine Wash Cold", output)
            self.assertNotEqual(output, REPLY)  # the frame itself must change

    def test_l2_output_differs_from_original(self):
        paraphraser = Paraphraser(level="L2", seed=0, cache_path="missing.jsonl")
        outputs = {paraphraser.paraphrase(REPLY, "sample_1", turn) for turn in range(1, 11)}
        self.assertTrue(all(output != REPLY for output in outputs))

    def test_l2_budget_rephrase_keeps_amount(self):
        message = "For that, what matters is: budget around $79.99."
        paraphraser = Paraphraser(level="L2", seed=0, cache_path="missing.jsonl")
        for turn in range(1, 11):
            self.assertIn("79.99", paraphraser.paraphrase(message, "sample_1", turn))

    def test_unrecognized_shape_untouched(self):
        paraphraser = Paraphraser(level="L2", seed=0, cache_path="missing.jsonl")
        self.assertEqual(paraphraser.paraphrase("Hi", "sample_1", 1), "Hi")

    def test_level_none_is_identity(self):
        paraphraser = Paraphraser(level="none", seed=0, cache_path="missing.jsonl")
        self.assertEqual(paraphraser.paraphrase(OPENING, "sample_1", 1), OPENING)

    def test_cache_replay_wins_over_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.jsonl"
            cache_path.write_text(
                json.dumps({"original": OPENING, "rewrite": "Cached rewrite."}) + "\n",
                encoding="utf-8",
            )
            paraphraser = Paraphraser(level="L1", seed=0, cache_path=cache_path)
            self.assertEqual(paraphraser.paraphrase(OPENING, "sample_1", 3), "Cached rewrite.")


if __name__ == "__main__":
    unittest.main()
