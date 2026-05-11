from dataclasses import dataclass


@dataclass
class SettlementCandidate:
    market_id: str
    outcome: str
