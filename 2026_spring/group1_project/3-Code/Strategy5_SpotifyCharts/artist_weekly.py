import pandas as pd
import requests
from datetime import date, timedelta
from io import StringIO

# ----------------------------
# Config
# ----------------------------
TOP_N = 100

# Past 7 completed days (exclude today)
today = date(2026, 2, 26)  # <-- set "today" to whatever you want
end_day = today - timedelta(days=1)
start_day = end_day - timedelta(days=4)  # 7 days inclusive: start..end

print(f"Date range: {start_day.isoformat()} → {end_day.isoformat()} (inclusive)")
print(f"Region: US | Top N: {TOP_N}")

# ----------------------------
# HTTP session
# ----------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
})

# ----------------------------
# Helpers
# ----------------------------
def assert_us_page(html: str, url: str):
    """
    Hard checks to ensure we really fetched the US daily chart page.
    Fail fast if we got unexpected content.
    """
    h = html.lower()

    # Basic sanity check that it's a Spotify daily chart page
    if "spotify daily chart" not in h:
        raise RuntimeError(f"Unexpected page content (missing 'Spotify Daily Chart') for {url}")

    # Stronger check: ensure the page references the country
    if "united states" not in h:
        raise RuntimeError(f"Page does not look like United States data (missing 'United States') for {url}")

def fetch_kworb_us_daily(d: date) -> pd.DataFrame:
    ds = d.strftime("%Y/%m/%d")
    url = f"https://kworb.net/spotify/country/us_daily.html?date={ds}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    html = resp.text

    # Verify it's USA page content
    assert_us_page(html, url)

    # Fixes FutureWarning: wrap literal html string
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise RuntimeError(f"No table found for {d} ({url})")

    df = tables[0]
    if "Artist and Title" not in df.columns or "Streams" not in df.columns:
        raise RuntimeError(f"Unexpected columns for {d}: {df.columns.tolist()} ({url})")

    return df

def to_int(x) -> int:
    return int(str(x).replace(",", "").strip())

def parse_artist(artist_and_title: str) -> str:
    # Kworb format is typically: "Artist - Title"
    parts = str(artist_and_title).split(" - ", 1)
    return parts[0].strip() if parts else str(artist_and_title).strip()

# ----------------------------
# Main: pull each day, keep Top N, aggregate
# ----------------------------
all_rows = []

d = start_day
while d <= end_day:
    df = fetch_kworb_us_daily(d).iloc[:TOP_N].copy()
    df["chart_date"] = d.isoformat()
    df["artist"] = df["Artist and Title"].map(parse_artist)
    df["streams"] = df["Streams"].map(to_int)

    all_rows.append(df[["chart_date", "artist", "streams", "Artist and Title"]])
    print("ok", d, "rows", len(df))
    d += timedelta(days=1)

week = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

artist_totals = (
    week.groupby("artist", as_index=False)["streams"]
        .sum()
        .sort_values("streams", ascending=False)
)

out_path = f"us_artist_streams_top{TOP_N}_{start_day}_{end_day}.csv"
artist_totals.to_csv(out_path, index=False)

print(f"\nWrote: {out_path}")
print("\nTop 20 artists by total streams (Top N only):")
print(artist_totals.head(20).to_string(index=False))