import io
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="SL News Trend Radar", layout="wide")
st.title("🇱🇰 Sri Lanka News Trend Radar")

# ---------- Load GOLD ----------
GOLD = Path("data/gold")
topics_path = GOLD / "topics_hourly.csv"
videos_path = GOLD / "videos_with_topics.csv"

if not topics_path.exists() or not videos_path.exists():
    st.error("Gold files not found. Run the jobs to generate:\n"
             "`python jobs/clean_and_features_pandas.py` → "
             "`python jobs/topic_cluster_aggregate_pandas.py`")
    st.stop()

topics = pd.read_csv(topics_path, parse_dates=["ts_hour"])
videos = pd.read_csv(videos_path, parse_dates=["ts_hour"])

# Backward-compat: optional columns may not exist
if "rising_score" not in topics.columns:
    topics["rising_score"] = (topics.get("zscore", 0).clip(lower=0, upper=5) / 5 * 100).round(1)
if "leader_channels" not in topics.columns:
    # compute quick leaders if not precomputed
    leaders = (videos.groupby(["ts_hour", "topic_id", "channelTitle"], as_index=False)
                    .agg(weight=("decayed_velocity", "sum")))
    leaders["rank"] = leaders.groupby(["ts_hour", "topic_id"])["weight"].rank(ascending=False, method="first")
    leaders_top = (leaders[leaders["rank"] <= 3]
                    .groupby(["ts_hour", "topic_id"], as_index=False)
                    .agg(leader_channels=("channelTitle", lambda s: ", ".join(s))))
    topics = topics.merge(leaders_top, on=["ts_hour","topic_id"], how="left")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Window")
    window_hours = st.selectbox("Hours", [1, 6, 12, 24], index=1)
    rising_only = st.checkbox("Rising only", value=True)
    min_videos = st.slider("Min videos in topic", 1, 10, 2)
    langs = sorted(videos["lang"].dropna().unique().tolist())
    lang_filter = st.multiselect("Languages", options=langs, default=langs)

# Apply time & rising filters
latest_ts = topics["ts_hour"].max()
cutoff = pd.to_datetime(latest_ts) - pd.Timedelta(hours=window_hours)

subset = topics[topics["ts_hour"] >= cutoff].copy()
if rising_only and "rising_flag" in subset.columns:
    subset = subset[subset["rising_flag"] == 1]

# Language filter: keep topics that have at least one video in selected langs
if lang_filter:
    topic_lang = (videos[videos["lang"].isin(lang_filter)]
                  .groupby("topic_id")["videoId"].nunique().reset_index(name="lang_vids"))
    subset = subset.merge(topic_lang, on="topic_id", how="left")
    subset = subset[subset["lang_vids"].fillna(0) > 0]

# Aggregate to Top topics for window
top = (subset.groupby("topic_id", as_index=False)
       .agg(velocity_sum=("velocity_sum","sum"),
            rising_score=("rising_score","mean"),
            video_count=("video_count","sum"),
            channel_count=("channel_count","sum"),
            sample_title=("sample_title","last"))
       .query("video_count >= @min_videos")
       .sort_values(["rising_score","velocity_sum"], ascending=[False, False])
       .head(20))

# ---------- KPI row ----------
col1, col2, col3 = st.columns(3)
col1.metric("Topics in window", f"{subset['topic_id'].nunique():,}")
col2.metric("Videos in window", f"{int(videos[videos['ts_hour']>=cutoff]['videoId'].nunique()):,}")
if not top.empty:
    col3.metric("Top rising score", f"{top['rising_score'].iloc[0]:.1f}")

# ---------- Top Topics ----------
st.subheader("Top Topics")

if top.empty:
    st.info("No topics match the current filters. Try widening the window or unchecking *Rising only*.")
else:
    # prettier table: add leader channels (last hour snapshot if present)
    latest_leaders = (subset.sort_values("ts_hour")
                      .drop_duplicates(["topic_id"], keep="last")[["topic_id","leader_channels"]])
    top = top.merge(latest_leaders, on="topic_id", how="left")

    # bar chart of top by velocity_sum
    fig = px.bar(top.head(10), x="velocity_sum", y="sample_title",
                 labels={"velocity_sum":"Momentum", "sample_title":"Topic"},
                 orientation="h", text="rising_score",
                 title="Top topics by momentum (text = rising score)")
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # styled table
    display = top[["topic_id","sample_title","leader_channels","rising_score","velocity_sum","video_count","channel_count"]]
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    # download button
    csv_buf = io.StringIO(); display.to_csv(csv_buf, index=False)
    st.download_button("Download top topics (CSV)", csv_buf.getvalue(), "top_topics.csv", "text/csv")

# ---------- Topic details ----------
st.subheader("Topic Details")
topic_options = top["topic_id"].tolist()
sel = st.selectbox("Select a topic", topic_options, index=0 if topic_options else None)

if topic_options:
    # trend over time
    hist = (subset[subset["topic_id"] == sel]
            .sort_values("ts_hour")[["ts_hour","velocity_sum","rising_score"]])
    if not hist.empty:
        fig2 = px.line(hist, x="ts_hour", y="velocity_sum", markers=True,
                       title="Momentum over time")
        st.plotly_chart(fig2, use_container_width=True)

    # channel contributions in window
    tv = videos[(videos["topic_id"] == sel) & (videos["ts_hour"] >= cutoff)]
    contrib = (tv.groupby("channelTitle", as_index=False)
                 .agg(contrib=("decayed_velocity","sum"))
                 .sort_values("contrib", ascending=False))
    if not contrib.empty:
        fig3 = px.bar(contrib.head(10), x="contrib", y="channelTitle",
                      orientation="h", title="Top contributing channels")
        fig3.update_layout(yaxis={'categoryorder':'total ascending'}, height=400, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    # quick search box for titles
    q = st.text_input("Filter titles (search)", "")
    t_videos = tv.sort_values("decayed_velocity", ascending=False)
    if q:
        q_lower = q.lower()
        t_videos = t_videos[t_videos["title_clean"].str.lower().str.contains(q_lower, na=False)]

    cols = ["ts_hour","channelTitle","title_clean","lang","decayed_velocity"]
    st.write(f"**Examples ({len(t_videos)})**")
    st.dataframe(t_videos[cols], use_container_width=True, hide_index=True)

    # download videos CSV
    vbuf = io.StringIO(); t_videos[cols].to_csv(vbuf, index=False)
    st.download_button("Download topic videos (CSV)", vbuf.getvalue(), "topic_videos.csv", "text/csv")
