import os, json, urllib.parse, urllib.request
from kalshi_common import load_private_key_pem, sign_pss_base64, now_ms

API_BASE = os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co")
V2 = "/trade-api/v2"

def headers(key_id, private_key, method, path):
    ts_ms = str(now_ms())
    path_to_sign = path.split("?", 1)[0]
    msg = ts_ms + method.upper() + path_to_sign
    sig = sign_pss_base64(private_key, msg)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }

def api_get(path, key_id, private_key):
    req = urllib.request.Request(API_BASE + path, headers=headers(key_id, private_key, "GET", path), method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def iter_paged(path_base, params, list_key, key_id, private_key):
    cursor = None
    while True:
        q = dict(params)
        if cursor:
            q["cursor"] = cursor
        path = f"{path_base}?{urllib.parse.urlencode(q)}"
        resp = api_get(path, key_id, private_key)
        items = resp.get(list_key, [])
        for it in items:
            yield it
        cursor = resp.get("cursor") or resp.get("next_cursor") or resp.get("next_page_token")
        if not cursor:
            return

def main():
    key_id = os.environ["KALSHI_KEY_ID"]
    private_key = load_private_key_pem(os.environ["KALSHI_PRIVATE_KEY_PEM"])

    # 1) Find candidate Olympics series inside Sports
    series_params = {"category": "Sports", "limit": "500"}
    olympic_series = []
    for s in iter_paged(f"{V2}/series", series_params, "series", key_id, private_key):
        title = (s.get("title") or "").lower()
        tags = [t.lower() for t in (s.get("tags") or [])]
        if "olympic" in title or any("olympic" in t for t in tags):
            olympic_series.append(s["ticker"])

    # 2) Pull all open markets for those series
    seen = set()
    all_open = []
    for st in olympic_series:
        mparams = {"series_ticker": st, "status": "open", "limit": "1000"}
        for m in iter_paged(f"{V2}/markets", mparams, "markets", key_id, private_key):
            t = m.get("ticker")
            if t and t not in seen:
                seen.add(t)
                all_open.append(m)

    import sys
    print(f"Found {len(all_open)} open Olympics-related markets across {len(olympic_series)} series.", file=sys.stderr)
    for m in all_open:
        print(json.dumps(m))



if __name__ == "__main__":
    main()
