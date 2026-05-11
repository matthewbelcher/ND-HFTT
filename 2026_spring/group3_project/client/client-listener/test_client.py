import argparse
import asyncio
import json
import random
import time
import uuid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TCP test client for client-listener")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--token", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--market-id", required=True)
    parser.add_argument("--count", type=int, default=5, help="number of orders to send")
    parser.add_argument(
        "--concurrency", type=int, default=3, help="number of concurrent send tasks"
    )
    return parser


def build_order(account_id: str, market_id: str) -> dict:
    return {
        "request_id": str(uuid.uuid4()),
        "account_id": account_id,
        "market_id": market_id,
        "side": random.choice(["YES", "NO"]),
        "qty": random.randint(1, 5),
        "price_cents": random.randint(10, 90),
        "time_in_force": "GTC",
        "ingress_ts_ns": time.time_ns(),
    }


async def send_submit_order(
    host: str,
    port: int,
    token: str,
    request_payload: dict,
) -> dict:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        message = {
            "action": "submit_order",
            "auth_token": token,
            "request": request_payload,
        }
        writer.write((json.dumps(message) + "\n").encode("utf-8"))
        await writer.drain()
        response_line = await reader.readline()
        if not response_line:
            return {"ok": False, "error": "no response from server"}
        return json.loads(response_line.decode("utf-8"))
    finally:
        writer.close()
        await writer.wait_closed()


async def run(args: argparse.Namespace) -> None:
    sem = asyncio.Semaphore(args.concurrency)

    async def _task(idx: int) -> None:
        async with sem:
            payload = build_order(args.account_id, args.market_id)
            resp = await send_submit_order(args.host, args.port, args.token, payload)
            print(f"[{idx}] request_id={payload['request_id']} response={resp}")

    await asyncio.gather(*[_task(i) for i in range(args.count)])


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    asyncio.run(run(cli_args))
