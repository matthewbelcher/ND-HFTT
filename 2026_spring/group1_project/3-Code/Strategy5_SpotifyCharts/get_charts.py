import pandas as pd
import requests
from datetime import date, timedelta

SONG_KEY = "Bad Bunny - DtMF"   # <-- exactly as it appears on the chart
YEAR = 2026
MONTH = 2

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
})

def fetch_us_daily_kworb(d: date) -> pd.DataFrame:
    ds = d.strftime("%Y/%m/%d")  # Kworb date param format
    url = f"https://kworb.net/spotify/country/us_daily.html?date={ds}"
    html = session.get(url, timeout=30).text
    tables = pd.read_html(html)
    if not tables:
        raise RuntimeError("No table found")
    return tables[0]

def parse_int(x) -> int:
    # handles "4,503,467" strings
    return int(str(x).replace(",", "").strip())

start = date(YEAR, MONTH, 1)
end = date(YEAR, MONTH + 1, 1) - timedelta(days=1) if MONTH < 12 else date(YEAR, 12, 31)

total = 0
daily_rows = []

d = start
while d <= end:
    df = fetch_us_daily_kworb(d)

    # Kworb daily table includes "Artist and Title" and "Streams" columns. :contentReference[oaicite:2]{index=2}
    if "Artist and Title" not in df.columns or "Streams" not in df.columns:
        raise RuntimeError(f"Unexpected columns on {d}: {df.columns.tolist()}")

    match = df[df["Artist and Title"] == SONG_KEY]
    streams = parse_int(match.iloc[0]["Streams"]) if len(match) else 0

    total += streams
    daily_rows.append({"date": d.isoformat(), "streams": streams})
    d += timedelta(days=1)

daily = pd.DataFrame(daily_rows)
print("Total streams in month:", total)
print(daily.head())
# daily.to_csv("daily_streams_feb.csv", index=False)