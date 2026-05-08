import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from normalize import normalize_orderbook


class TestNormalize(unittest.TestCase):
    def test_yes_no_bids_to_asks(self) -> None:
        ob = {"orderbook": {"yes": [[40, 5]], "no": [[60, 7]]}}
        book = normalize_orderbook("MKT", ob)
        self.assertEqual(book.yes_bid, (40, 5))
        self.assertEqual(book.no_bid, (60, 7))
        self.assertEqual(book.yes_ask, (40, 7))
        self.assertEqual(book.no_ask, (60, 5))

    def test_missing_no_book(self) -> None:
        ob = {"orderbook": {"yes": [[30, 2]]}}
        book = normalize_orderbook("MKT", ob)
        self.assertIsNone(book.yes_ask)
        self.assertEqual(book.no_ask, (70, 2))

    def test_fp_dollars_fallback(self) -> None:
        ob = {"orderbook": {}, "orderbook_fp": {"yes_dollars": [[0.41, 3]]}}
        book = normalize_orderbook("MKT", ob)
        self.assertEqual(book.yes_bid, (41, 3))


if __name__ == "__main__":
    unittest.main()
