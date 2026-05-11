import argparse
import asyncio
import json
import time
import uuid


def _build_message(args: argparse.Namespace, request_id: str) -> bytes:
    payload = {
        "action": "submit_order",
        "auth_token": args.token,
        "request": {
            "request_id": request_id,
            "account_id": args.account_id,
            "market_id": args.market_id,
            "side": args.side,
            "qty": args.qty,
            "price_cents": args.price_cents,
            "time_in_force": args.time_in_force,
            "ingress_ts_ns": time.time_ns(),
        },
    }
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


async def send_once(args: argparse.Namespace, request_id: str) -> dict:
    reader, writer = await asyncio.open_connection(args.host, args.port)
    try:
        writer.write(_build_message(args, request_id))
        await writer.drain()
        line = await reader.readline()
        return json.loads(line.decode("utf-8"))
    finally:
        writer.close()
        await writer.wait_closed()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Submit orders to client-listener TCP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--token", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--market-id", required=True)
    parser.add_argument("--side", choices=["YES", "NO"], default="YES")
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument("--price-cents", type=int, default=60)
    parser.add_argument("--time-in-force", choices=["GTC", "IOC", "FOK"], default="GTC")
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()

    for _ in range(args.count):
        request_id = str(uuid.uuid4())
        response = await send_once(args, request_id)
        print(f"request_id={request_id} response={response}")


if __name__ == "__main__":
    asyncio.run(main())
