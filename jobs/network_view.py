import pandas as pd
from pathlib import Path

GOLD = Path("data/gold")
videos = pd.read_csv(GOLD / "videos_with_topics.csv", parse_dates=["ts_hour"])

# Edge list: channel ↔ topic (weighted by sum of decayed_velocity)
edges = (videos.groupby(["channelTitle","topic_id"], as_index=False)
               .agg(weight=("decayed_velocity","sum")))

# Channel centrality proxy: sum of weights over topics
channel_centrality = edges.groupby("channelTitle", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
channel_centrality.to_csv(GOLD / "network_channel_centrality.csv", index=False)

# Topic “degree” proxy: sum of weights over channels
topic_strength = edges.groupby("topic_id", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
topic_strength.to_csv(GOLD / "network_topic_strength.csv", index=False)

print("Wrote:", GOLD / "network_channel_centrality.csv", "and", GOLD / "network_topic_strength.csv")
