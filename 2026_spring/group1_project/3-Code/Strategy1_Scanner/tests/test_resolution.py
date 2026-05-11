from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCANNER_DIR = Path(__file__).resolve().parents[1]
if str(SCANNER_DIR) not in sys.path:
    sys.path.insert(0, str(SCANNER_DIR))

from full_set_arb import guaranteed_no_payout, guaranteed_yes_payout
from resolution import infer_resolution_spec


class ResolutionInferenceTests(unittest.TestCase):
    def test_mutually_exclusive_is_one_yes(self) -> None:
        spec = infer_resolution_spec(
            {"event_ticker": "KXTEST", "mutually_exclusive": True},
            [{"ticker": "A"}, {"ticker": "B"}],
        )
        self.assertEqual(spec.yes_count, 1)
        self.assertEqual(spec.source, "kalshi_mutually_exclusive")

    def test_config_override_wins_for_series(self) -> None:
        spec = infer_resolution_spec(
            {"event_ticker": "KXF1TOP10-MIAGP26", "series_ticker": "KXF1TOP10"},
            [{"ticker": str(i)} for i in range(20)],
            config={"series": {"KXF1TOP10": {"yes_count": 10}}},
        )
        self.assertEqual(spec.yes_count, 10)
        self.assertEqual(spec.source, "config")

    def test_top_n_inferred_from_title_and_rules(self) -> None:
        spec = infer_resolution_spec(
            {
                "event_ticker": "KXF1TOP10-MIAGP26",
                "title": "Miami Grand Prix: Top 10 Finishers",
                "mutually_exclusive": False,
            },
            [
                {
                    "ticker": f"KXF1TOP10-MIAGP26-{i}",
                    "rules_primary": "If this driver finishes in the top 10, then the market resolves to Yes.",
                }
                for i in range(20)
            ],
        )
        self.assertEqual(spec.yes_count, 10)
        self.assertEqual(spec.source, "top_n_inferred")

    def test_conflicting_top_n_is_not_exact(self) -> None:
        spec = infer_resolution_spec(
            {
                "event_ticker": "KXTOP5-TEST",
                "title": "Top 10 Finishers",
                "mutually_exclusive": False,
            },
            [{"ticker": str(i)} for i in range(20)],
        )
        self.assertFalse(spec.is_exact)
        self.assertEqual(spec.source, "conflict")

    def test_medal_event_inferred_as_three_yes(self) -> None:
        spec = infer_resolution_spec(
            {
                "event_ticker": "KXWOFREESKI-XTAER26MEDAL",
                "title": "Freestyle Skiing Mixed Team Aerials: Medal Winner",
                "mutually_exclusive": False,
            },
            [{"ticker": str(i), "title": "Will a team win any medal?"} for i in range(7)],
        )
        self.assertEqual(spec.yes_count, 3)
        self.assertEqual(spec.source, "medal_inferred")

    def test_gold_medal_market_is_not_inferred_as_three(self) -> None:
        spec = infer_resolution_spec(
            {
                "event_ticker": "KXTEST-GOLDMEDAL",
                "title": "Gold Medal Winner",
                "mutually_exclusive": False,
            },
            [{"ticker": str(i), "title": "Will a team win the gold medal?"} for i in range(7)],
        )
        self.assertFalse(spec.is_exact)


class GuaranteedPayoutTests(unittest.TestCase):
    def test_full_top_ten_coverage_pays_ten_each_side_with_twenty_markets(self) -> None:
        self.assertEqual(guaranteed_yes_payout(10, total_markets=20, covered_markets=20), 10.0)
        self.assertEqual(guaranteed_no_payout(10, covered_markets=20), 10.0)

    def test_partial_yes_coverage_uses_worst_case_lower_bound(self) -> None:
        self.assertEqual(guaranteed_yes_payout(10, total_markets=20, covered_markets=19), 9.0)
        self.assertEqual(guaranteed_yes_payout(1, total_markets=20, covered_markets=19), 0.0)

    def test_partial_no_coverage_uses_worst_case_lower_bound(self) -> None:
        self.assertEqual(guaranteed_no_payout(10, covered_markets=19), 9.0)
        self.assertEqual(guaranteed_no_payout(1, covered_markets=19), 18.0)


if __name__ == "__main__":
    unittest.main()
