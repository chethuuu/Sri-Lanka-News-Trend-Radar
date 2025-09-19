import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# GOLD = Path("data/gold")

st.set_page_config(page_title="SL News Trend Radar", layout="wide")
st.title("🇱🇰 Sri Lanka News Trend Radar")

# import pandas as pd
# from pathlib import Path

GOLD = Path("data/gold")
topics_path = GOLD / "topics_hourly.csv"
videos_path = GOLD / "videos_with_topics.csv"

topics = pd.read_csv(topics_path, parse_dates=["ts_hour"])
videos = pd.read_csv(videos_path, parse_dates=["ts_hour"])

# Controls
window_hours = st.sidebar.selectbox("Window", [1, 6, 12, 24], index=1)
rising_only = st.sidebar.checkbox("Rising only", value=True)

latest_ts = topics["ts_hour"].max()
cutoff = pd.to_datetime(latest_ts) - pd.Timedelta(hours=window_hours)

subset = topics[topics["ts_hour"] >= cutoff]
if rising_only:
    subset = subset[subset["rising_flag"] == 1]

top = (subset.groupby("topic_id", as_index=False)
       .agg(velocity_sum=("velocity_sum","sum"),
            video_count=("video_count","sum"),
            channel_count=("channel_count","sum"),
            sample_title=("sample_title","last"))
       .sort_values("velocity_sum", ascending=False)
       .head(20))

st.subheader("Top Topics")
st.dataframe(top[["topic_id","sample_title","velocity_sum","video_count","channel_count"]])

st.subheader("Topic Details")
sel = st.selectbox("Select a topic", top["topic_id"].tolist() if len(top) else [])
if sel != "":
    t_videos = videos[videos["topic_id"] == sel].sort_values("decayed_velocity", ascending=False).head(50)
    st.write(f"**Examples ({len(t_videos)})**")
    st.dataframe(t_videos[["ts_hour","channelTitle","title_clean","lang","decayed_velocity"]])
