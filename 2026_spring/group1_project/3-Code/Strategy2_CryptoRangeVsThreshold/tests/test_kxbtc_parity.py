import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from kxbtc_parity_math import (
    Bucket,
    Threshold,
    compute_constant_payout_trades,
    compute_bucket_from_thresholds,
    compute_threshold_from_buckets,
    negative_implied_bucket_check,
)
from kxbtc_parity_parse import parse_range_text, parse_threshold_text


class TestKXBtcParityParsing(unittest.TestCase):
    def test_parse_range_between(self) -> None:
        r = parse_range_text("Between $45,000 and $46,000")
        self.assertIsNotNone(r)
        assert r
        self.assertEqual(r.raw_low, 45000.0)
        self.assertEqual(r.raw_high, 46000.0)
        self.assertEqual(r.norm_low, 45000)
        self.assertEqual(r.norm_high, 46000)
        self.assertIsNone(r.decimal_warning)

    def test_parse_range_dash(self) -> None:
        r = parse_range_text("45000-46000")
        self.assertIsNotNone(r)
        assert r
        self.assertEqual(r.norm_low, 45000)
        self.assertEqual(r.norm_high, 46000)

    def test_parse_range_decimal_warning(self) -> None:
        r = parse_range_text("Between 44999.99 and 45000.00")
        self.assertIsNotNone(r)
        assert r
        self.assertIsNotNone(r.decimal_warning)
        self.assertIsNone(r.norm_low)
        self.assertEqual(r.norm_high, 45000)

    def test_parse_threshold_ge(self) -> None:
        t = parse_threshold_text("BTC at or above 45,000")
        self.assertIsNotNone(t)
        assert t
        self.assertEqual(t.raw_strike, 45000.0)
        self.assertEqual(t.norm_strike, 45000)

    def test_parse_threshold_reject_below(self) -> None:
        t = parse_threshold_text("BTC below 45,000")
        self.assertIsNone(t)


class TestKXBtcParityMath(unittest.TestCase):
    def test_threshold_from_buckets_alignment(self) -> None:
        buckets = [
            Bucket("B0", 0.0, 10.0, 0, 10, 10.0),
            Bucket("B1", 10.0, 20.0, 10, 20, 20.0),
            Bucket("B2", 20.0, 30.0, 20, 30, 30.0),
        ]
        thresholds = [
            Threshold("T10", 10.0, 10, 50.0),
            Threshold("T20", 20.0, 20, 30.0),
        ]
        implied, trades = compute_threshold_from_buckets(
            buckets, thresholds, True, 0.5
        )
        self.assertEqual(implied[0].implied_yes_geK, 50.0)
        self.assertEqual(implied[1].implied_yes_geK, 30.0)
        self.assertEqual(len(trades), 0)

    def test_bucket_from_thresholds_identity(self) -> None:
        buckets = [
            Bucket("B0", 0.0, 10.0, 0, 10, 10.0),
            Bucket("B1", 10.0, 20.0, 10, 20, 20.0),
            Bucket("B2", 20.0, 30.0, 20, 30, 30.0),
        ]
        thresholds = [
            Threshold("T0", 0.0, 0, 60.0),
            Threshold("T10", 10.0, 10, 50.0),
            Threshold("T20", 20.0, 20, 30.0),
            Threshold("T30", 30.0, 30, 0.0),
        ]
        implied, trades = compute_bucket_from_thresholds(buckets, thresholds, 0.5)
        match = {row.low: row for row in implied}
        self.assertAlmostEqual(match[10].implied_bucket, 20.0)
        self.assertAlmostEqual(match[20].implied_bucket, 30.0)
        self.assertEqual(len(trades), 0)

    def test_partial_bounds(self) -> None:
        buckets = [
            Bucket("B0", 0.0, 10.0, 0, 10, 10.0),
            Bucket("B1", 10.0, 20.0, 10, 20, 20.0),
            Bucket("B2", 20.0, 30.0, 20, 30, 30.0),
        ]
        thresholds = [Threshold("T15", 15.0, None, 40.0)]
        implied, trades = compute_threshold_from_buckets(
            buckets, thresholds, True, 1.0
        )
        row = implied[0]
        self.assertEqual(row.lower_bound, 30.0)
        self.assertEqual(row.upper_bound, 50.0)
        self.assertTrue(row.partial_bucket_risk)
        self.assertEqual(len(trades), 0)

    def test_negative_implied_bucket(self) -> None:
        thresholds = [
            Threshold("T10", 10.0, 10, 30.0),
            Threshold("T20", 20.0, 20, 40.0),
        ]
        alerts = negative_implied_bucket_check(thresholds, 0.5)
        self.assertEqual(len(alerts), 1)

    def test_constant_payout_trade(self) -> None:
        buckets = [
            Bucket("B10", 10.0, 20.0, 10, 20, 5.0),
            Bucket("B20", 20.0, 30.0, 20, 30, 5.0),
        ]
        thresholds = [
            Threshold("T10", 10.0, 10, 60.0),
            Threshold("T30", 30.0, 30, 0.0),
        ]
        trades = compute_constant_payout_trades(
            buckets, thresholds, min_payout_dollars=2, max_payout_dollars=5
        )
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade.payout_per_contract, 300)
        self.assertEqual(len(trade.legs), 4)


if __name__ == "__main__":
    unittest.main()
