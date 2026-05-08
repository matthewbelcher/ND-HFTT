import base64
import time
from typing import Dict, Optional, Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"


def now_ms() -> int:
    return int(time.time() * 1000)


# ----- Auth (Kalshi RSA signing) -----
def load_private_key_pem(pem_path: str) -> rsa.RSAPrivateKey:
    with open(pem_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def sign_pss_base64(private_key: rsa.RSAPrivateKey, message: str) -> str:
    sig = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


def kalshi_ws_headers(key_id: str, pem_path: str) -> Dict[str, str]:
    ts_ms = str(now_ms())
    private_key = load_private_key_pem(pem_path)
    msg = ts_ms + "GET" + WS_PATH
    sig = sign_pss_base64(private_key, msg)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }


# ----- Tiny book for YES bids and NO bids -----
class Book:
    def __init__(self) -> None:
        self.yes: Dict[int, int] = {}  # price_cents -> qty
        self.no: Dict[int, int] = {}   # price_cents -> qty

    def apply_snapshot(self, yes_levels, no_levels) -> None:
        self.yes = {int(p): int(q) for p, q in yes_levels if int(q) > 0}
        self.no = {int(p): int(q) for p, q in no_levels if int(q) > 0}

    def apply_delta(self, side: str, price: int, delta: int) -> None:
        book = self.yes if side == "yes" else self.no
        new_qty = book.get(price, 0) + delta
        if new_qty <= 0:
            book.pop(price, None)
        else:
            book[price] = new_qty

    def best_yes_bid(self) -> Optional[Tuple[int, int]]:
        if not self.yes:
            return None
        p = max(self.yes)
        return p, self.yes[p]

    def best_no_bid(self) -> Optional[Tuple[int, int]]:
        if not self.no:
            return None
        p = max(self.no)
        return p, self.no[p]

    def yes_bbo(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns YES (bid_px, bid_qty, ask_px, ask_qty) in cents.
        YES ask is implied by best NO bid:
            YES_ask_px = 100 - NO_bid_px
            YES_ask_qty = NO_bid_qty
        """
        yb = self.best_yes_bid()
        nb = self.best_no_bid()
        if not yb or not nb:
            return None
        bid_px, bid_qty = yb
        ask_px = 100 - nb[0]
        ask_qty = nb[1]
        return bid_px, bid_qty, ask_px, ask_qty
