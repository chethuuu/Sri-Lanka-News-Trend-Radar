# jobs/topic_cluster_aggregate_pandas.py
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

SILVER = Path("data/silver")
GOLD = Path("data/gold")
GOLD.mkdir(parents=True, exist_ok=True)

def main():
    # Ensure datetimes are parsed
    df = pd.read_csv(SILVER / "youtube_clean.csv")
    if df.empty:
        print("No data in silver.")
        return

    # Robust date parsing (handles strings in CSV)
    for col in ["publishedAt", "fetchedAt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Drop rows without fetchedAt/publishedAt
    df = df.dropna(subset=["fetchedAt", "publishedAt"]).copy()
    if df.empty:
        print("No valid rows after datetime parsing.")
        return

    # ---- Text features & clustering (TF-IDF + KMeans) ----
    texts = df["title_clean"].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
    X = tfidf.fit_transform(texts)

    k = min(40, max(2, len(df) // 50))  # sensible k if dataset is small
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["topic_id"] = km.fit_predict(X)

    # Hour bucket (local-ish via UTC hour; good enough for assignment)
    df["ts_hour"] = df["fetchedAt"].dt.floor("H")

    # ----- Leader channels per topic/hour -----
    leaders = (df.groupby(["ts_hour", "topic_id", "channelTitle"], as_index=False)
                 .agg(weight=("decayed_velocity", "sum")))
    leaders["rank"] = leaders.groupby(["ts_hour", "topic_id"])["weight"] \
                             .rank(ascending=False, method="first")
    leaders_top = (leaders[leaders["rank"] <= 3]
                   .groupby(["ts_hour", "topic_id"], as_index=False)
                   .agg(leader_channels=("channelTitle", lambda s: ", ".join(s))))

    # ----- Hourly topic aggregates -----
    agg = (df.groupby(["ts_hour", "topic_id"], as_index=False)
             .agg(velocity_sum=("decayed_velocity", "sum"),
                  video_count=("videoId", "nunique"),
                  channel_count=("channelId", "nunique"),
                  sample_title=("title_clean", "last")))

    agg = agg.merge(leaders_top, on=["ts_hour", "topic_id"], how="left")

    # Rolling baseline per topic (expanding mean/std)
    agg = agg.sort_values(["topic_id", "ts_hour"]).copy()
    agg["vel_mean"] = agg.groupby("topic_id")["velocity_sum"] \
                         .transform(lambda s: s.shift(1).expanding().mean())
    agg["vel_std"] = agg.groupby("topic_id")["velocity_sum"] \
                        .transform(lambda s: s.shift(1).expanding().std()) \
                        .fillna(1.0)

    agg["zscore"] = (agg["velocity_sum"] - agg["vel_mean"].fillna(0.0)) / \
                    agg["vel_std"].replace(0, 1)

    # Nice 0–100 “rising score” for UI
    agg["rising_flag"] = (agg["zscore"] >= 2.0).astype(int)
    agg["rising_score"] = (agg["zscore"].clip(lower=0, upper=5) / 5.0 * 100).round(1)

    # Save GOLD
    agg.to_csv(GOLD / "topics_hourly.csv", index=False)
    df[["videoId", "channelId", "channelTitle", "title_clean", "lang",
        "decayed_velocity", "ts_hour", "topic_id"]].to_csv(GOLD / "videos_with_topics.csv", index=False)

    print("Wrote gold/topics_hourly.csv and gold/videos_with_topics.csv")

if __name__ == "__main__":
    main()
