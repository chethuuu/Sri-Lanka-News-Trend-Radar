# jobs/topic_cluster_aggregate_pandas.py
"""
Silver -> Gold:
- Cluster titles into topics (TF-IDF + KMeans).
- Build robust human-friendly topic labels (Sinhala/Tamil/English safe).
- Aggregate momentum per topic per hour.
- Compute rising score (z-score vs expanding baseline).
- Save:
    data/gold/topics_hourly.csv
    data/gold/videos_with_topics.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# --- Unicode-aware tools (keeps Sinhala/Tamil combining marks intact) ---
import regex as re           # pip install regex
import unicodedata as ud

# Accept any sequence of letters (L) + marks (M) + numbers (N)
WORD_RE = re.compile(r"[\p{L}\p{M}\p{N}]+", flags=re.UNICODE)

SILVER = Path("data/silver")
GOLD = Path("data/gold")
GOLD.mkdir(parents=True, exist_ok=True)


def extract_words(text: str):
    """Normalize to NFC and return words consisting of letters/marks/numbers."""
    if not isinstance(text, str):
        return []
    text = ud.normalize("NFC", text)
    return WORD_RE.findall(text)


def make_topic_label(title_series: pd.Series, topk: int = 3) -> str:
    """
    Build a short, human-friendly label from the most frequent bigrams
    in the topic's titles. Falls back to frequent unigrams if needed.
    Works for Sinhala/Tamil/English because we keep combining marks.
    """
    toks = []
    rows = title_series.dropna().astype(str).head(300)  # cap for speed
    for t in rows:
        words = extract_words(t)
        toks += [" ".join(p) for p in zip(words, words[1:])]

    # Fallback to unigrams if no bigrams were formed
    if not toks:
        uni = []
        for t in rows:
            uni += extract_words(t)
        common_uni = [w for w, _ in Counter(uni).most_common(topk)]
        return ", ".join(common_uni)

    common_bi = [w for w, _ in Counter(toks).most_common(topk)]
    return ", ".join(common_bi)


def readable_label(label: str, sample_title: str) -> str:
    """
    Guardrail: if a label degenerates into mostly single-char tokens,
    fall back to a real example title (or leave label if OK).
    """
    parts = [p for p in label.split() if p]
    if not parts:
        return sample_title or label
    short = sum(1 for p in parts if len(p) <= 1)
    if len(parts) and short / len(parts) > 0.4:
        return sample_title or label
    return label


def main():
    # ---- Load Silver ----
    silver_csv = SILVER / "youtube_clean.csv"
    if not silver_csv.exists():
        print("No silver file found. Run the cleaner first.")
        return

    df = pd.read_csv(silver_csv)
    if df.empty:
        print("No data in silver.")
        return

    # Parse timestamps as UTC datetimes (robust to strings)
    for col in ["publishedAt", "fetchedAt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Keep only rows with valid timestamps
    df = df.dropna(subset=["fetchedAt", "publishedAt"]).copy()
    if df.empty:
        print("No valid rows after datetime parsing.")
        return

    # ---- Text features & clustering (TF-IDF + KMeans) ----
    texts = df["title_clean"].fillna("").astype(str)

    # Safer params: adapt to dataset size
    n_docs = len(df)
    max_feats = 12000 if n_docs > 500 else 8000
    min_df = 2 if n_docs >= 10 else 1
    ngram = (1, 2) if n_docs >= 20 else (1, 1)

    tfidf = TfidfVectorizer(max_features=max_feats, ngram_range=ngram, min_df=min_df)
    X = tfidf.fit_transform(texts)

    # Adaptive number of clusters:
    #  - at least 4, at most 60; roughly one topic per ~80 docs
    k = min(60, max(4, n_docs // 80)) if n_docs >= 40 else max(2, n_docs // 5)
    k = int(max(2, min(k, n_docs)))  # never exceed #docs

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["topic_id"] = km.fit_predict(X)

    # ---- Hour bucket (UTC) ----
    df["ts_hour"] = df["fetchedAt"].dt.floor("H")

    # ---- Optional: Short, human-friendly labels ----
    labels = (
        df.groupby("topic_id")["title_clean"]
          .apply(make_topic_label)
          .reset_index()
          .rename(columns={"title_clean": "topic_label"})
    )
    df = df.merge(labels, on="topic_id", how="left")

    # ---- Leader channels per topic/hour (sum of decayed velocity) ----
    leaders = (
        df.groupby(["ts_hour", "topic_id", "channelTitle"], as_index=False)
          .agg(weight=("decayed_velocity", "sum"))
    )
    leaders["rank"] = leaders.groupby(["ts_hour", "topic_id"])["weight"] \
                             .rank(ascending=False, method="first")
    leaders_top = (
        leaders[leaders["rank"] <= 3]
        .groupby(["ts_hour", "topic_id"], as_index=False)
        .agg(leader_channels=("channelTitle", lambda s: ", ".join(s)))
    )

    # ---- Hourly topic aggregates ----
    agg = (
        df.groupby(["ts_hour", "topic_id"], as_index=False)
          .agg(
              velocity_sum=("decayed_velocity", "sum"),
              video_count=("videoId", "nunique"),
              channel_count=("channelId", "nunique"),
              sample_title=("title_clean", "last"),
          )
    )
    # Add leaders + labels
    agg = agg.merge(leaders_top, on=["ts_hour", "topic_id"], how="left")
    agg = agg.merge(labels, on="topic_id", how="left")

    # Final pass to ensure labels are readable
    agg["topic_label"] = [
        readable_label(lbl or "", samp or "")
        for lbl, samp in zip(agg.get("topic_label", ""), agg.get("sample_title", ""))
    ]

    # ---- Rising detection vs expanding baseline per topic ----
    agg = agg.sort_values(["topic_id", "ts_hour"]).copy()
    # expanding mean/std of prior hours (shift(1) avoids peeking at current)
    agg["vel_mean"] = agg.groupby("topic_id")["velocity_sum"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    agg["vel_std"] = agg.groupby("topic_id")["velocity_sum"].transform(
        lambda s: s.shift(1).expanding().std()
    ).fillna(1.0)

    agg["zscore"] = (agg["velocity_sum"] - agg["vel_mean"].fillna(0.0)) / agg["vel_std"].replace(0, 1)

    # Map z-score to a 0–100 "rising score" for the UI
    agg["rising_flag"] = (agg["zscore"] >= 2.0).astype(int)
    agg["rising_score"] = (agg["zscore"].clip(lower=0, upper=5) / 5.0 * 100).round(1)

    # ---- Save GOLD ----
    topics_cols = [
        "ts_hour", "topic_id", "topic_label", "leader_channels",
        "velocity_sum", "video_count", "channel_count",
        "zscore", "rising_flag", "rising_score", "sample_title"
    ]
    (GOLD / "topics_hourly.csv").write_text(
        agg[topics_cols].to_csv(index=False), encoding="utf-8"
    )

    # Keep platform if present (for multi-source UI filters)
    video_cols = [
        "videoId", "channelId", "channelTitle", "title_clean", "lang",
        "decayed_velocity", "ts_hour", "topic_id"
    ]
    if "platform" in df.columns:
        video_cols.append("platform")

    (GOLD / "videos_with_topics.csv").write_text(
        df[video_cols].to_csv(index=False), encoding="utf-8"
    )

    print(
        "Wrote gold/topics_hourly.csv and gold/videos_with_topics.csv | "
        f"topics={agg['topic_id'].nunique()} | videos={len(df)} | k={k}"
    )


if __name__ == "__main__":
    main()
