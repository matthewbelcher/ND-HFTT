from dataclasses import dataclass


@dataclass
class Order:
    request_id: str
    global_seq: int
    account_id: str
    market_id: str
    side: str
    remaining_qty: int
    price_cents: int
    status: str
