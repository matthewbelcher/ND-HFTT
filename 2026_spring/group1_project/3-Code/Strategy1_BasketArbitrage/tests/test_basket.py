import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from basket import FeeModel, evaluate_basket, simulate_buy


class TestBasket(unittest.TestCase):
    def test_fee_cents_taker(self) -> None:
        fee = FeeModel("taker", taker_rate=0.07)
        self.assertEqual(fee.fee_cents(50, 1), 2)

    def test_simulate_buy(self) -> None:
        fee = FeeModel("none")
        levels = [(40, 2), (45, 2)]
        cost, filled = simulate_buy(levels, 3, fee)
        self.assertEqual(filled, 3)
        self.assertEqual(cost, 40 * 2 + 45 * 1)

    def test_evaluate_basket_yes(self) -> None:
        fee = FeeModel("none")
        levels = [[(40, 10)], [(45, 10)]]
        result = evaluate_basket(
            "yes",
            levels,
            target_qty=1,
            payout_per_contract=100,
            fee_model=fee,
        )
        self.assertEqual(result.filled_qty, 1)
        self.assertEqual(result.cost_cents, 85)
        self.assertEqual(result.edge_cents, 15)


if __name__ == "__main__":
    unittest.main()
