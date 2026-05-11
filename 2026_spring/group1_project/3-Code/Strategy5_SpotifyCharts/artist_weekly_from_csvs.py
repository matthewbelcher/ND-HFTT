import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# ----------------------------
# Config
# ----------------------------
DISPLAY_TOP_N = 50

CHART_DIR = Path(r"C:\Users\jackd\OneDrive\Documents\VSCodeWork\KalshiProject\Spotify\ChartCSVs")

start_day = date(2026, 1, 30)
end_day = date(2026, 2, 5)

# How to credit collabs:
# "fractional" (recommended) splits streams evenly across credited artists
# "full" credits full streams to every credited artist (inflates totals)
CREDIT_MODE = "fractional"  # "fractional" or "full"

ARTIST_SPLIT_RE = re.compile(
    r"\s*(?:,|&|\+|\band\b|\bwith\b|\bfeat\.?\b|\bft\.?\b|\bfeaturing\b)\s*|\s+\bx\b\s+",
    re.IGNORECASE,
)

# Find "feat/ft/featuring/with ..." anywhere in the title, including inside parentheses/brackets
TRACK_FEAT_ANYWHERE_RE = re.compile(
    r"(?:\(|\[)?\s*(?:feat\.?|ft\.?|featuring|with)\s+([^\)\]]+)\s*(?:\)|\])?",
    re.IGNORECASE,
)

FILE_RE = re.compile(r"regional-us-daily-(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)

print(f"Date range: {start_day.isoformat()} -> {end_day.isoformat()} (inclusive)")
print(f"Region: US | Display Top N: {DISPLAY_TOP_N} | Source dir: {CHART_DIR}")
print(f"Credit mode: {CREDIT_MODE}")

# ----------------------------
# Helpers
# ----------------------------
def list_files_in_range(chart_dir: Path, start: date, end: date):
    files = []
    for path in chart_dir.glob("regional-us-daily-*.csv"):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        d = date.fromisoformat(m.group(1))
        if start <= d <= end:
            files.append((d, path))
    files.sort(key=lambda x: x[0])
    return files

def to_int_series(s: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(
            s.astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

def normalize_artist(a: str) -> str:
    a = str(a).strip()
    a = re.sub(r"\s+", " ", a)
    return a

def split_artists(s: str) -> list[str]:
    parts = [normalize_artist(x) for x in ARTIST_SPLIT_RE.split(str(s)) if str(x).strip()]
    # de-dupe while preserving order
    seen = set()
    out = []
    for p in parts:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out

def extract_track_features(title: str) -> list[str]:
    if not isinstance(title, str):
        return []
    feats = []
    for m in TRACK_FEAT_ANYWHERE_RE.finditer(title):
        feat_str = m.group(1).strip()
        if feat_str:
            feats.extend(split_artists(feat_str))
    return feats

def load_daily(path: Path, chart_date: date) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"artist_names", "streams"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns {sorted(missing)} in {path}")

    df = df.copy()
    df["chart_date"] = chart_date.isoformat()
    df["streams"] = to_int_series(df["streams"])
    df["artist_names"] = df["artist_names"].astype(str)

    # Optional rank handling for extra features
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(9999).astype(int)
    else:
        df["rank"] = 9999

    # Build artist lists from artist slot + featured artists from track_name
    df["artist_list"] = df["artist_names"].apply(split_artists)

    if "track_name" in df.columns:
        df["artist_list"] = df.apply(
            lambda r: r["artist_list"] + extract_track_features(r.get("track_name", "")),
            axis=1,
        )
    else:
        df["track_name"] = ""

    # Clean + de-dupe per row
    df["artist_list"] = df["artist_list"].apply(lambda xs: [x for x in xs if x])

    # Compute credit per artist
    df["n_artists"] = df["artist_list"].apply(lambda xs: max(len(xs), 1))
    if CREDIT_MODE == "fractional":
        df["streams_credit"] = (df["streams"] / df["n_artists"]).astype(float)
    elif CREDIT_MODE == "full":
        df["streams_credit"] = df["streams"].astype(float)
    else:
        raise ValueError("CREDIT_MODE must be 'fractional' or 'full'")

    # Explode for aggregation
    df = df.explode("artist_list")
    df["artist"] = df["artist_list"].map(normalize_artist)
    df = df[df["artist"] != ""]

    # Track key for breadth metrics
    track_key = (
        df.get("spotify_id", pd.Series([""] * len(df)))
        if "spotify_id" in df.columns
        else (df["artist_names"].astype(str) + " - " + df["track_name"].astype(str))
    )
    df["track_key"] = track_key.astype(str)

    # Rank-weighted streams (emphasize top ranks)
    df["rank_weighted"] = df["streams_credit"] / df["rank"].clip(lower=1)

    return df[["chart_date", "artist", "streams_credit", "rank_weighted", "track_key"]]

# ----------------------------
# Main
# ----------------------------
files = list_files_in_range(CHART_DIR, start_day, end_day)
if not files:
    raise RuntimeError(f"No CSV files found in range {start_day}..{end_day} in {CHART_DIR}")

expected_days = []
d = start_day
while d <= end_day:
    expected_days.append(d)
    d += timedelta(days=1)

found_days = {d for d, _ in files}
missing_days = [d for d in expected_days if d not in found_days]
if missing_days:
    print("Warning: missing files for:", ", ".join(d.isoformat() for d in missing_days))

all_rows = []
for chart_date, path in files:
    day_df = load_daily(path, chart_date)
    all_rows.append(day_df)
    print("ok", chart_date, "rows", len(day_df))

week = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

artist_features = (
    week.groupby("artist", as_index=False)
        .agg(
            streams=("streams_credit", "sum"),
            rank_weighted=("rank_weighted", "sum"),
            num_tracks=("track_key", pd.Series.nunique),
            num_days=("chart_date", pd.Series.nunique),
        )
        .sort_values("rank_weighted", ascending=False)
)

out_dir = Path(__file__).resolve().parent
out_path = out_dir / f"us_artist_features_{start_day}_{end_day}_from_csvs.csv"
artist_features.to_csv(out_path, index=False)

print(f"\nWrote: {out_path}")
print(f"\nTop {DISPLAY_TOP_N} artists (by rank_weighted):")
print(artist_features.head(DISPLAY_TOP_N).to_string(index=False))
