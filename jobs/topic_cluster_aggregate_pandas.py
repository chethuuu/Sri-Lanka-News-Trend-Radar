import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

SILVER = Path("data/silver")
GOLD = Path("data/gold")
GOLD.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(SILVER / "youtube_clean.csv", parse_dates=["publishedAt","fetchedAt"])
    if df.empty:
        print("No data in silver.")
        return

    # Simple TF-IDF (no torch needed)
    texts = df["title_clean"].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2), min_df=2)
    X = tfidf.fit_transform(texts)

    k = 40  # adjust if needed
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["topic_id"] = km.fit_predict(X)

    # Hour bucket in local time
    df["ts_hour"] = df["fetchedAt"].dt.floor("H")

    # Hourly topic aggregates
    agg = (df.groupby(["ts_hour","topic_id"])
             .agg(velocity_sum=("decayed_velocity","sum"),
                  video_count=("videoId","nunique"),
                  channel_count=("channelId","nunique"),
                  sample_title=("title_clean","last"))
             .reset_index())

    # Rolling baseline per topic (14 days ~ 336 hours). Use expanding mean/std as a proxy.
    agg = agg.sort_values(["topic_id","ts_hour"])
    agg["vel_mean"] = agg.groupby("topic_id")["velocity_sum"].transform(lambda s: s.shift(1).expanding().mean())
    agg["vel_std"]  = agg.groupby("topic_id")["velocity_sum"].transform(lambda s: s.shift(1).expanding().std()).fillna(1.0)
    agg["zscore"]   = (agg["velocity_sum"] - agg["vel_mean"].fillna(0.0)) / agg["vel_std"].replace(0,1)
    agg["rising_flag"] = (agg["zscore"] >= 2.0).astype(int)

    # Save GOLD
    agg.to_csv(GOLD / "topics_hourly.csv", index=False)
    df[["videoId","channelId","channelTitle","title_clean","lang","decayed_velocity","ts_hour","topic_id"]] \
      .to_csv(GOLD / "videos_with_topics.csv", index=False)

    print("Wrote gold/topics_hourly.csv and gold/videos_with_topics.csv")

if __name__ == "__main__":
    main()
