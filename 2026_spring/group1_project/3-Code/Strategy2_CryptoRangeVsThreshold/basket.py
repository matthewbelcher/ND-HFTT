from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

PriceLevel = Tuple[int, int]


class FeeModel:
    def __init__(self, mode: str, taker_rate: float = 0.07, maker_rate: float = 0.0) -> None:
        mode = (mode or "none").lower()
        if mode not in {"taker", "maker", "none"}:
            raise ValueError(f"Invalid fee mode: {mode}")
        self.mode = mode
        self.taker_rate = float(taker_rate)
        self.maker_rate = float(maker_rate)

    @property
    def rate(self) -> float:
        if self.mode == "taker":
            return self.taker_rate
        if self.mode == "maker":
            return self.maker_rate
        return 0.0

    def fee_cents(self, price_cents: int, contracts: int) -> int:
        rate = self.rate
        if rate <= 0 or price_cents <= 0 or contracts <= 0:
            return 0
        p = price_cents / 100.0
        fee_dollars = rate * contracts * p * (1.0 - p)
        return int(math.ceil(fee_dollars * 100.0 - 1e-9))

    def buy_cost_cents(self, price_cents: int, contracts: int) -> int:
        return price_cents * contracts + self.fee_cents(price_cents, contracts)

    def sell_proceeds_cents(self, price_cents: int, contracts: int) -> int:
        return price_cents * contracts - self.fee_cents(price_cents, contracts)


@dataclass
class BasketEvaluation:
    side: str
    legs: int
    target_qty: int
    filled_qty: int
    max_qty_top: int
    max_qty_depth: int
    cost_cents: int
    payout_cents: int
    edge_cents: int
    roi: Optional[float]


def _sum_qty(levels: Iterable[PriceLevel]) -> int:
    return sum(q for _, q in levels if q > 0)


def max_qty_top(level_sets: List[List[PriceLevel]]) -> int:
    if not level_sets:
        return 0
    qtys: List[int] = []
    for levels in level_sets:
        if not levels:
            return 0
        qtys.append(levels[0][1])
    return min(qtys) if qtys else 0


def max_qty_depth(level_sets: List[List[PriceLevel]]) -> int:
    if not level_sets:
        return 0
    qtys: List[int] = []
    for levels in level_sets:
        if not levels:
            return 0
        qtys.append(_sum_qty(levels))
    return min(qtys) if qtys else 0


def simulate_buy(levels: List[PriceLevel], qty: int, fee_model: FeeModel) -> Tuple[int, int]:
    if qty <= 0:
        return 0, 0
    remaining = qty
    cost = 0
    filled = 0
    for price, level_qty in levels:
        if remaining <= 0:
            break
        take = min(remaining, level_qty)
        if take <= 0:
            continue
        cost += fee_model.buy_cost_cents(price, take)
        filled += take
        remaining -= take
    return cost, filled


def basket_cost_for_qty(
    level_sets: List[List[PriceLevel]],
    qty: int,
    fee_model: FeeModel,
) -> Tuple[int, int]:
    if qty <= 0 or not level_sets:
        return 0, 0
    max_depth_qty = max_qty_depth(level_sets)
    fill_qty = min(qty, max_depth_qty)
    if fill_qty <= 0:
        return 0, 0
    total_cost = 0
    for levels in level_sets:
        cost, filled = simulate_buy(levels, fill_qty, fee_model)
        if filled < fill_qty:
            return 0, 0
        total_cost += cost
    return total_cost, fill_qty


def evaluate_basket(
    side: str,
    level_sets: List[List[PriceLevel]],
    target_qty: int,
    payout_per_contract: int,
    fee_model: FeeModel,
) -> BasketEvaluation:
    legs = len(level_sets)
    top_qty = max_qty_top(level_sets)
    depth_qty = max_qty_depth(level_sets)
    cost, filled_qty = basket_cost_for_qty(level_sets, target_qty, fee_model)
    payout = payout_per_contract * filled_qty
    edge = payout - cost
    roi = None
    if cost > 0:
        roi = edge / cost
    return BasketEvaluation(
        side=side,
        legs=legs,
        target_qty=target_qty,
        filled_qty=filled_qty,
        max_qty_top=top_qty,
        max_qty_depth=depth_qty,
        cost_cents=cost,
        payout_cents=payout,
        edge_cents=edge,
        roi=roi,
    )
