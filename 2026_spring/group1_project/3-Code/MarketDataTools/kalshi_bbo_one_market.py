import os, json, time, base64, asyncio
from typing import Dict, Optional, Tuple

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"


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
    ts_ms = str(int(time.time() * 1000))
    private_key = load_private_key_pem(pem_path)
    msg = ts_ms + "GET" + WS_PATH  # per Kalshi docs
    sig = sign_pss_base64(private_key, msg)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }


# ----- Tiny book for YES bids and NO bids -----
class Book:
    def __init__(self):
        self.yes: Dict[int, int] = {}  # price_cents -> qty
        self.no: Dict[int, int] = {}   # price_cents -> qty

    def apply_snapshot(self, yes_levels, no_levels):
        self.yes = {int(p): int(q) for p, q in yes_levels if int(q) > 0}
        self.no  = {int(p): int(q) for p, q in no_levels  if int(q) > 0}

    def apply_delta(self, side: str, price: int, delta: int):
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


async def main():
    market_ticker = "KXWOMHOCKEY-26FEB14DEULVA-DEU"

    key_id = os.environ["KALSHI_KEY_ID"]
    pem_path = os.environ["KALSHI_PRIVATE_KEY_PEM"]
    headers = kalshi_ws_headers(key_id, pem_path)

    book = Book()
    async with websockets.connect(
    WS_URL,
    additional_headers=headers,
    ping_interval=20,
    ping_timeout=20,
    max_size=2**23,
    )as ws:
        sub = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": [market_ticker],
            },
        }
        await ws.send(json.dumps(sub))
        print(f"Subscribed: {market_ticker}")

        while True:
            msg = json.loads(await ws.recv())
            mtype = msg.get("type")
            payload = msg.get("msg", {})

            if payload.get("market_ticker") != market_ticker:
                continue

            if mtype == "orderbook_snapshot":
                book.apply_snapshot(payload.get("yes", []), payload.get("no", []))

            elif mtype == "orderbook_delta":
                side = payload["side"]         # "yes" or "no"
                price = int(payload["price"])  # cents
                delta = int(payload["delta"])  # qty change
                book.apply_delta(side, price, delta)

            else:
                continue

            bbo = book.yes_bbo()
            if not bbo:
                continue

            bid_px, bid_qty, ask_px, ask_qty = bbo
            spread = ask_px - bid_px
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] YES {bid_qty} @ {bid_px:02d} | {ask_qty} @ {ask_px:02d}  spread={spread}c")


if __name__ == "__main__":
    asyncio.run(main())
