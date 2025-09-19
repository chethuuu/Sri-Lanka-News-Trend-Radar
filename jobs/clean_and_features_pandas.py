# jobs/clean_and_features_pandas.py

import numpy as np
import pandas as pd
from pathlib import Path
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42

BRONZE = Path("data/bronze")
SILVER = Path("data/silver")
SILVER.mkdir(parents=True, exist_ok=True)


def detect_lang_safe(text: str):
    try:
        if not text:
            return None
        return detect(text)
    except Exception:
        return None


def get_video_id(x):
    """
    Handles both shapes:
    - videos().list -> id is a string
    - search().list -> id is a dict with {"videoId": "..."}
    """
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("videoId")
    return None


def g(d, k):
    """Null-safe dict getter."""
    return d.get(k) if isinstance(d, dict) else None


def main():
    files = sorted(BRONZE.glob("youtube_raw_*.jsonl"))
    if not files:
        print("No bronze files found. Run ingest first.")
        return

    # Load & concat raw JSONL dumps
    dfs = [pd.read_json(f, lines=True) for f in files]
    raw = pd.concat(dfs, ignore_index=True)
    print("Raw rows:", len(raw))

    # Flatten nested fields (null-safe)
    df = pd.DataFrame({
        "videoId": raw["id"].apply(get_video_id),
        "title": raw["snippet"].apply(lambda s: g(s, "title")),
        "description": raw["snippet"].apply(lambda s: g(s, "description")),
        "channelId": raw["snippet"].apply(lambda s: g(s, "channelId")),
        "channelTitle": raw["snippet"].apply(lambda s: g(s, "channelTitle")),
        "publishedAt": pd.to_datetime(raw["snippet"].apply(lambda s: g(s, "publishedAt")), errors="coerce"),
        "fetchedAt": pd.to_datetime(raw["_fetched_at"], errors="coerce"),
        "views": pd.to_numeric(raw["statistics"].apply(lambda s: g(s, "viewCount")), errors="coerce"),
        "likes": pd.to_numeric(raw["statistics"].apply(lambda s: g(s, "likeCount")), errors="coerce"),
        "comments": pd.to_numeric(raw["statistics"].apply(lambda s: g(s, "commentCount")), errors="coerce"),
    })

    print("After flatten:", len(df))

    # Drop rows missing key fields
    df = df.dropna(subset=["videoId", "publishedAt", "fetchedAt"])
    if df.empty:
        print("No valid rows after dropping missing keys (videoId/publishedAt/fetchedAt).")
        out = SILVER / "youtube_clean.csv"
        df.to_csv(out, index=False)
        print("Wrote", out)
        return

    # Deduplicate: keep the latest snapshot per videoId
    df = df.sort_values(["videoId", "fetchedAt"]).drop_duplicates(subset=["videoId"], keep="last")

    # Clean text & detect language
    df["title_clean"] = (
        df["title"].fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["lang"] = df["title_clean"].apply(detect_lang_safe)

    # Fill metrics nulls with 0
    for col in ["views", "likes", "comments"]:
        df[col] = df[col].fillna(0).astype(float)

    # Age in minutes (non-negative)
    age_min = (df["fetchedAt"] - df["publishedAt"]).dt.total_seconds() / 60.0
    df["age_min"] = np.clip(age_min.values, 0.0001, None)

    # Rates per minute
    df["views_per_min"] = df["views"] / df["age_min"]
    df["likes_per_min"] = df["likes"] / df["age_min"]
    df["comments_per_min"] = df["comments"] / df["age_min"]

    # Recency-decayed velocity (τ = 120 min)
    tau = 120.0
    df["decayed_velocity"] = df["views_per_min"] * np.exp(-df["age_min"] / tau)

    # Persist SILVER
    out = SILVER / "youtube_clean.csv"
    df.to_csv(out, index=False)
    print("Wrote", out, "| rows:", len(df))


if __name__ == "__main__":
    main()
