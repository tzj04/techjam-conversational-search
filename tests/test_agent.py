from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import ALLOWED_ATTRIBUTES, Agent


def write_catalog(root: Path) -> Path:
    rows = [
        {
            "parent_asin": "A",
            "title": "Black leather hiking boot",
            "features": ["waterproof traction sole", "comfortable ankle support"],
            "details": {"department": "womens", "material": "leather"},
            "description": ["rugged winter hiking boot"],
            "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Boots"],
            "store": "TrailCo",
            "average_rating": 4.8,
            "rating_number": 200,
            "price": 84.0,
        },
        {
            "parent_asin": "B",
            "title": "Pink cotton summer dress",
            "features": ["lightweight floral style"],
            "details": {"department": "womens", "material": "cotton"},
            "description": ["casual beach dress"],
            "categories": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Dresses"],
            "store": "Sunny",
            "average_rating": 4.4,
            "rating_number": 100,
            "price": 32.0,
        },
        {
            "parent_asin": "C",
            "title": "Blue running sneaker",
            "features": ["breathable mesh", "wide fit"],
            "details": {"department": "mens", "material": "mesh"},
            "description": ["training shoe for gym running"],
            "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Sneakers"],
            "store": "RunFast",
            "average_rating": 4.6,
            "rating_number": 180,
            "price": 59.0,
        },
    ]
    path = root / "catalog.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class AgentPrototypeTest(unittest.TestCase):
    def test_recommends_valid_unique_catalog_ids_and_asks_allowed_attribute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(write_catalog(Path(directory)))
            agent.reset("s1", {"summary": "Prior purchases emphasize comfort.", "preference_tags": ["comfort"]})

            response = agent.respond(
                "s1",
                "I'm looking for boots. A key requirement is: leather.",
                1,
                10,
            )

            ids = [item["parent_asin"] for item in response["recommendations"]]
            self.assertEqual(ids[0], "A")
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(set(ids) <= {"A", "B", "C"})
            self.assertIn(response["ask_attribute"], ALLOWED_ATTRIBUTES | {None})

    def test_sessions_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(write_catalog(Path(directory)))
            profile = {"summary": "", "preference_tags": []}
            agent.reset("boots", profile)
            agent.reset("dress", profile)

            boot_response = agent.respond("boots", "I'm looking for boots. leather.", 1, 10)
            dress_response = agent.respond("dress", "I'm looking for dresses. pink cotton.", 1, 10)

            self.assertEqual(boot_response["recommendations"][0]["parent_asin"], "A")
            self.assertEqual(dress_response["recommendations"][0]["parent_asin"], "B")

    def test_override_drops_stale_preference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(write_catalog(Path(directory)))
            agent.reset("s1", {"summary": "", "preference_tags": []})

            agent.respond("s1", "I'm looking for shoes. pink dress", 1, 10)
            response = agent.respond(
                "s1",
                "Actually, ignore my earlier preference. What I need is: blue running.",
                2,
                10,
            )

            self.assertEqual(response["recommendations"][0]["parent_asin"], "C")

    def test_no_preference_reply_does_not_filter_everything(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(write_catalog(Path(directory)))
            agent.reset("s1", {"summary": "", "preference_tags": []})

            first = agent.respond("s1", "I'm looking for boots, but I'm still exploring.", 1, 10)
            response = agent.respond(
                "s1",
                f"I don't have a preference for {first['ask_attribute']}; please use your judgment.",
                2,
                10,
            )

            self.assertTrue(response["recommendations"])
            self.assertEqual(response["recommendations"][0]["parent_asin"], "A")


if __name__ == "__main__":
    unittest.main()
