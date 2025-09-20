"""
Bronze → Silver cleaner for the Sri Lanka News Trend Radar.

What it does
------------
- Loads Bronze JSONL dumps (YouTube and optional Google Trends).
- Flattens nested fields into a tabular schema.
- Computes features: age (minutes), per-minute rates, recency-decayed momentum.
- Best-effort language detection (robust to errors / short titles).
- Deduplicates on (platform, videoId) keeping the latest snapshot.
- Writes Silver as CSV (easy) and Parquet (fast, compact).
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from langdetect import detect, DetectorFactory

# Make langdetect deterministic
DetectorFactory.seed = 42

# Medallion paths
BRONZE = Path("data/bronze")
SILVER = Path("data/silver")
SILVER.mkdir(parents=True, exist_ok=True)

# ---------- Utilities ----------

def detect_lang_safe(text: str):
    """Best-effort language detection for non-empty strings."""
    try:
        if not text:
            return None
        t = text.strip()
        if len(t) < 3:
            return None
        return detect(t)
    except Exception:
        return None

def get_video_id(x):
    """
    Normalize ID field shape:
    - videos().list → id is a string
    - search().list → id is a dict with {"videoId": "..."}
    """
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("videoId")
    return None

def g(d, k, default=None):
    """Null-safe dict getter: returns d[k] if d is a dict, else default."""
    return d.get(k, default) if isinstance(d, dict) else default

_ISO8601_DURATION = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)

def parse_iso8601_duration_to_seconds(s: str):
    """PT#H#M#S → seconds (returns None if missing/invalid)."""
    if not isinstance(s, str):
        return None
    m = _ISO8601_DURATION.match(s)
    if not m:
        return None
    days = int(m.group("days") or 0)
    hours = int(m.group("hours") or 0)
    minutes = int(m.group("minutes") or 0)
    seconds = int(m.group("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds

def load_bronze_frames():
    """
    Read Bronze JSONL files:
      - youtube_raw_*.jsonl  (from API/yt-dlp)
      - trends_raw_*.jsonl   (optional)
    Returns a single concatenated DataFrame or None.
    """
    files = sorted(BRONZE.glob("youtube_raw_*.jsonl")) + \
            sorted(BRONZE.glob("trends_raw_*.jsonl"))
    if not files:
        print("No bronze files found. Run an ingest job first.")
        return None
    dfs = [pd.read_json(f, lines=True) for f in files]
    raw = pd.concat(dfs, ignore_index=True)
    print(f"Bronze files: {len(files)} | Raw rows: {len(raw):,}")
    return raw

# ---------- Main ----------

def main():
    raw = load_bronze_frames()
    if raw is None or raw.empty:
        out = SILVER / "youtube_clean.csv"
        pd.DataFrame().to_csv(out, index=False)
        print("Wrote empty", out)
        return

    # Normalize / default platform
    if "platform" in raw.columns:
        platform_series = raw["platform"].fillna("youtube").astype(str).str.lower()
    else:
        platform_series = pd.Series(["youtube"] * len(raw))

    # Title fallback: use snippet.localized.title if snippet.title is missing
    def pick_title(s):
        if not isinstance(s, dict):
            return None
        return s.get("title") or g(s.get("localized"), "title")

    # Duration (optional) from contentDetails.duration
    duration_sec = raw.get("contentDetails").apply(
        lambda c: parse_iso8601_duration_to_seconds(g(c, "duration"))
    ) if "contentDetails" in raw.columns else None

    # Flatten into a uniform schema
    df = pd.DataFrame({
        "platform": platform_series,
        "videoId": raw["id"].apply(get_video_id),
        "title": raw["snippet"].apply(pick_title),
        "description": raw["snippet"].apply(lambda s: g(s, "description")),
        "channelId": raw["snippet"].apply(lambda s: g(s, "channelId")),
        "channelTitle": raw["snippet"].apply(lambda s: g(s, "channelTitle")),
        "publishedAt": pd.to_datetime(
            raw["snippet"].apply(lambda s: g(s, "publishedAt")),
            errors="coerce", utc=True
        ),
        "fetchedAt": pd.to_datetime(raw["_fetched_at"], errors="coerce", utc=True),
        "views": pd.to_numeric(raw["statistics"].apply(lambda s: g(s, "viewCount")), errors="coerce"),
        "likes": pd.to_numeric(raw["statistics"].apply(lambda s: g(s, "likeCount")), errors="coerce"),
        "comments": pd.to_numeric(raw["statistics"].apply(lambda s: g(s, "commentCount")), errors="coerce"),
    })

    if duration_sec is not None:
        df["duration_sec"] = duration_sec

    print("After flatten:", len(df))

    # Keep rows with essential keys
    df = df.dropna(subset=["videoId", "publishedAt", "fetchedAt"])
    if df.empty:
        out = SILVER / "youtube_clean.csv"
        df.to_csv(out, index=False)
        print("No valid rows after key-field drop; wrote", out)
        return

    # Fix any “future publish” anomalies (clock skew): cap publishedAt at fetchedAt
    mask_future = df["publishedAt"] > df["fetchedAt"]
    if mask_future.any():
        df.loc[mask_future, "publishedAt"] = df.loc[mask_future, "fetchedAt"]

    # Deduplicate by (platform, videoId), keep latest snapshot
    df = (df.sort_values(["platform", "videoId", "fetchedAt"])
            .drop_duplicates(subset=["platform", "videoId"], keep="last"))

    # Clean text & detect language
    df["title_clean"] = (
        df["title"].fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["lang"] = df["title_clean"].apply(detect_lang_safe)

    # Fill numeric nulls with 0
    for col in ["views", "likes", "comments"]:
        df[col] = df[col].fillna(0).astype(float)

    # Age in minutes (non-negative)
    age_min = (df["fetchedAt"] - df["publishedAt"]).dt.total_seconds() / 60.0
    df["age_min"] = np.clip(age_min.values, 0.0001, None)

    # Per-minute rates
    df["views_per_min"] = df["views"] / df["age_min"]
    df["likes_per_min"] = df["likes"] / df["age_min"]
    df["comments_per_min"] = df["comments"] / df["age_min"]

    # Clip extreme outliers (optional but makes plots saner)
    for col in ["views_per_min", "likes_per_min", "comments_per_min"]:
        hi = df[col].quantile(0.999)  # top 0.1% cap
        if pd.notna(hi) and hi > 0:
            df[col] = np.clip(df[col], 0, hi)

    # Recency-decayed momentum (τ = 120 min)
    tau = 120.0
    df["decayed_velocity"] = df["views_per_min"] * np.exp(-df["age_min"] / tau)

    # Enforce column order (nice for downstream / grading)
    cols = [
        "platform", "videoId", "channelId", "channelTitle",
        "title", "title_clean", "description", "lang",
        "publishedAt", "fetchedAt",
        "views", "likes", "comments",
        "age_min", "views_per_min", "likes_per_min", "comments_per_min",
        "decayed_velocity"
    ]
    if "duration_sec" in df.columns:
        cols.insert(8, "duration_sec")  # after title/desc fields
    df = df[cols]

    # Persist Silver
    out_csv = SILVER / "youtube_clean.csv"
    out_parq = SILVER / "youtube_clean.parquet"
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parq, index=False)
    except Exception:
        pass  # Parquet is optional

    # Simple summary for logs / viva
    by_plat = df.groupby("platform").size().reset_index(name="rows")
    print("Wrote", out_csv, "| rows:", len(df))
    print("Rows by platform:")
    for _, r in by_plat.iterrows():
        print(f"  - {r['platform']}: {int(r['rows']):,}")

if __name__ == "__main__":
    main()
