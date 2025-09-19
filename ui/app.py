# ui/app.py
# 🇱🇰 Sri Lanka News Trend Radar — Streamlit UI (with robust CSV error handling)
# - Safe CSV loader (falls back to Python engine, skips bad lines, can read Spark part-*.csv)
# - Shows how many malformed lines were skipped
# - Local time (Asia/Colombo), Last-N-hours or Custom date range
# - Platform & language filters, Channel Leaderboard
# - Clickable YouTube links in examples
# - “Repair & save” button to rewrite clean CSVs

import io
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import plotly.express as px

# -------- Streamlit setup --------
st.set_page_config(page_title="SL News Trend Radar", layout="wide")
st.title("🇱🇰 Sri Lanka News Trend Radar")

# -------- Paths --------
GOLD = Path("data/gold")
TOPICS_CSV = GOLD / "topics_hourly.csv"
VIDEOS_CSV = GOLD / "videos_with_topics.csv"

if not TOPICS_CSV.exists() or not VIDEOS_CSV.exists():
    st.error(
        "Gold files not found. Run the jobs to generate:\n"
        "• `python jobs/clean_and_features_pandas.py`\n"
        "• `python jobs/topic_cluster_aggregate_pandas.py`\n"
        "  (or Spark variants, then copy part files to single CSVs)"
    )
    st.stop()

# -------- Robust CSV loader --------
def _detect_encoding(path: Path) -> str:
    """Best-effort encoding detection; defaults to utf-8 if chardet missing/fails."""
    try:
        import chardet  # optional dep
        with open(path, "rb") as f:
            raw = f.read(4096)
        enc = chardet.detect(raw or b"")["encoding"] or "utf-8"
        return enc
    except Exception:
        return "utf-8"

def _read_csv_tolerant(path: Path, date_cols):
    """
    1) Try fast C engine
    2) Fallback to Python engine + on_bad_lines='skip'
    3) If single CSV missing/invalid, try Spark folder 'path_dir/part-*.csv'
    Returns (DataFrame, skipped_bad_lines:int, source_path:str)
    """
    skipped = 0
    used_path = str(path)
    enc = _detect_encoding(path) if path.exists() else "utf-8"

    def _count_lines(p: Path, encoding="utf-8"):
        try:
            with open(p, "r", encoding=encoding, errors="ignore") as f:
                return sum(1 for _ in f)
        except Exception:
            return None

    # --- 1) Fast path
    try:
        df = pd.read_csv(path, parse_dates=date_cols, encoding=enc)
    except Exception:
        # --- 2) Tolerant path
        try:
            df = pd.read_csv(
                path, parse_dates=date_cols,
                engine="python", on_bad_lines="skip",
                encoding=enc, quotechar='"'
            )
            total = _count_lines(path, enc)
            if total is not None:
                skipped = max(total - len(df) - 1, 0)  # minus header
        except Exception:
            # --- 3) Try Spark part file
            folder = Path(str(path) + "_dir")
            parts = sorted(glob.glob(str(folder / "part-*.csv")))
            if not parts:
                raise  # Nothing else to try
            used_path = parts[0]
            df = pd.read_csv(
                used_path, parse_dates=date_cols,
                engine="python", on_bad_lines="skip",
                encoding="utf-8", quotechar='"'
            )
            total = _count_lines(Path(used_path), "utf-8")
            if total is not None:
                skipped = max(total - len(df) - 1, 0)

    # Remove duplicate header rows if any (rare concatenation artifact)
    if not df.empty and list(df.columns) == list(df.iloc[0].values):
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = df.columns.str.strip()
    return df, skipped, used_path

@st.cache_data(show_spinner=False)
def load_gold(topics_path: Path, videos_path: Path):
    topics, t_skipped, t_src = _read_csv_tolerant(topics_path, ["ts_hour"])
    videos, v_skipped, v_src = _read_csv_tolerant(videos_path, ["ts_hour"])

    # dtype safety for topic_id
    if "topic_id" in topics.columns:
        topics["topic_id"] = pd.to_numeric(topics["topic_id"], errors="coerce").astype("Int64")
    if "topic_id" in videos.columns:
        videos["topic_id"] = pd.to_numeric(videos["topic_id"], errors="coerce").astype("Int64")

    return topics, videos, t_skipped, v_skipped, t_src, v_src

topics, videos, t_skipped, v_skipped, t_src, v_src = load_gold(TOPICS_CSV, VIDEOS_CSV)

if topics.empty or videos.empty:
    st.warning("Gold files are empty after parsing. Please collect & process data first.")
    st.stop()

# -------- Show loader diagnostics if any --------
with st.expander("Load diagnostics", expanded=False):
    st.write(f"Topics source: `{t_src}` | Skipped bad lines: **{t_skipped}**")
    st.write(f"Videos source: `{v_src}` | Skipped bad lines: **{v_skipped}**")

if t_skipped or v_skipped:
    st.warning(
        f"Loaded with minor repairs: {t_skipped} bad line(s) in topics, "
        f"{v_skipped} in videos were skipped. Displayed data omits those malformed rows."
    )

# -------- Local time columns --------
LK = pytz.timezone("Asia/Colombo")
topics["ts_hour"] = pd.to_datetime(topics["ts_hour"], utc=True)
videos["ts_hour"] = pd.to_datetime(videos["ts_hour"], utc=True)
topics["ts_hour_local"] = topics["ts_hour"].dt.tz_convert(LK)
videos["ts_hour_local"] = videos["ts_hour"].dt.tz_convert(LK)

# -------- Backward-compat fields --------
if "rising_score" not in topics.columns:
    topics["rising_score"] = (topics.get("zscore", 0).clip(lower=0, upper=5) / 5 * 100).round(1)
if "leader_channels" not in topics.columns:
    leaders = (
        videos.groupby(["ts_hour", "topic_id", "channelTitle"], as_index=False)
              .agg(weight=("decayed_velocity", "sum"))
    )
    leaders["rank"] = leaders.groupby(["ts_hour", "topic_id"])["weight"] \
                             .rank(ascending=False, method="first")
    leaders_top = (leaders[leaders["rank"] <= 3]
                   .groupby(["ts_hour", "topic_id"], as_index=False)
                   .agg(leader_channels=("channelTitle", lambda s: ", ".join(s))))
    topics = topics.merge(leaders_top, on=["ts_hour", "topic_id"], how="left")

# -------- Sidebar: Window / Filters + Maintenance --------
with st.sidebar:
    st.header("Window")
    mode = st.radio("Mode", ["Last N hours", "Custom range"], horizontal=True)
    if mode == "Last N hours":
        window_hours = st.selectbox("Hours", [1, 6, 12, 24], index=1)
        latest_ts = topics["ts_hour"].max()
        cutoff_utc = pd.to_datetime(latest_ts) - pd.Timedelta(hours=window_hours)
        time_mask = topics["ts_hour"] >= cutoff_utc
    else:
        min_local = topics["ts_hour_local"].min().date()
        max_local = topics["ts_hour_local"].max().date()
        d1, d2 = st.date_input("Date range (Colombo time)", value=(min_local, max_local))
        # Protect against odd return types
        start_utc = pd.Timestamp(d1, tz=LK).to_utc_datetime().tz_localize(None)
        end_utc = (pd.Timestamp(d2, tz=LK) + pd.Timedelta(days=1)).to_utc_datetime().tz_localize(None)
        time_mask = (topics["ts_hour"] >= start_utc) & (topics["ts_hour"] < end_utc)

    rising_only = st.checkbox("Rising only", value=True)
    min_videos = st.slider("Min videos in topic", 1, 10, 2)

    langs = sorted(videos["lang"].dropna().unique().tolist())
    lang_filter = st.multiselect("Languages", options=langs, default=langs)

    if "platform" in videos.columns:
        platforms = sorted(videos["platform"].fillna("youtube").unique().tolist())
        plat_filter = st.multiselect("Platforms", options=platforms, default=platforms)
    else:
        plat_filter = None

    with st.expander("Maintenance"):
        if st.button("🛠 Repair & save clean CSVs"):
            try:
                topics.to_csv(TOPICS_CSV, index=False)
                videos.to_csv(VIDEOS_CSV, index=False)
                st.success("Rewrote clean CSVs. (You may need to refresh the page.)")
            except Exception as e:
                st.error(f"Failed to save: {e}")

# -------- Apply filters --------
subset = topics[time_mask].copy()
if rising_only and "rising_flag" in subset.columns:
    subset = subset[subset["rising_flag"] == 1]

# Language filter: keep topics that have at least one video in selected langs
if lang_filter:
    topic_lang = (videos[videos["lang"].isin(lang_filter)]
                  .groupby("topic_id")["videoId"].nunique().reset_index(name="lang_vids"))
    subset = subset.merge(topic_lang, on="topic_id", how="left")
    subset = subset[subset["lang_vids"].fillna(0) > 0]

# Platform filter
if plat_filter is not None and len(plat_filter) > 0:
    vids_in_plat = videos[videos.get("platform", "youtube").isin(plat_filter)]
    topic_ids_keep = vids_in_plat["topic_id"].dropna().unique()
    subset = subset[subset["topic_id"].isin(topic_ids_keep)]

# Prefer topic_label
if "topic_label" not in subset.columns:
    subset["topic_label"] = subset.get("sample_title", "")

# -------- Aggregate Top topics --------
top = (subset.groupby("topic_id", as_index=False)
       .agg(velocity_sum=("velocity_sum","sum"),
            rising_score=("rising_score","mean"),
            video_count=("video_count","sum"),
            channel_count=("channel_count","sum"),
            sample_title=("sample_title","last"),
            topic_label=("topic_label","last"))
       .query("video_count >= @min_videos")
       .sort_values(["rising_score","velocity_sum"], ascending=[False, False])
       .head(20))

# -------- KPIs --------
k1, k2, k3 = st.columns(3)
k1.metric("Topics in window", f"{subset['topic_id'].nunique():,}")
videos_in_window = videos[videos["ts_hour"].isin(subset["ts_hour"].unique())]
k2.metric("Videos in window", f"{int(videos_in_window['videoId'].nunique()):,}")
k3.metric("Top rising score", f"{top['rising_score'].iloc[0]:.1f}" if not top.empty else "—")

# -------- Tabs --------
tab_topics, tab_details, tab_leader = st.tabs(["Top Topics", "Topic Details", "Channel Leaderboard"])

# === Top Topics ===
with tab_topics:
    st.subheader("Top Topics")
    if top.empty:
        st.info("No topics match the current filters. Try widening the window or unchecking *Rising only*.")
    else:
        latest_leaders = (subset.sort_values("ts_hour")
                          .drop_duplicates(["topic_id"], keep="last")[["topic_id","leader_channels"]])
        top_disp = top.merge(latest_leaders, on="topic_id", how="left")

        label_series = top_disp["topic_label"].fillna("").replace("", pd.NA).fillna(top_disp["sample_title"])
        label_for_chart = label_series.str.slice(0, 80)

        fig = px.bar(
            top_disp.head(10),
            x="velocity_sum",
            y=label_for_chart.head(10),
            labels={"velocity_sum":"Momentum","y":"Topic"},
            orientation="h",
            text="rising_score",
            title="Top topics by momentum (text = rising score)",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=500,
            margin=dict(l=10, r=10, t=60, b=10),
        )
        fig.update_traces(hovertemplate="Momentum=%{x:.2f}<br>Rising=%{text}")
        st.plotly_chart(fig, use_container_width=True)

        table = top_disp[["topic_id","topic_label","sample_title","leader_channels",
                          "rising_score","velocity_sum","video_count","channel_count"]]
        st.dataframe(table, use_container_width=True, hide_index=True)

        csv_buf = io.StringIO(); table.to_csv(csv_buf, index=False)
        st.download_button("Download top topics (CSV)", csv_buf.getvalue(), "top_topics.csv", "text/csv")

# === Topic Details ===
with tab_details:
    st.subheader("Topic Details")
    topic_options = [int(x) for x in top["topic_id"].dropna().astype(int).tolist()] if not top.empty else []
    sel = st.selectbox("Select a topic", topic_options, index=0 if topic_options else None)

    if topic_options:
        hist = (subset[subset["topic_id"] == sel]
                .sort_values("ts_hour")[["ts_hour","ts_hour_local","velocity_sum","rising_score"]])
        if not hist.empty:
            fig2 = px.line(hist, x="ts_hour_local", y="velocity_sum", markers=True,
                           title="Momentum over time (Colombo)")
            fig2.update_layout(margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        # Videos in selected topic within time window
        topic_ts = subset[subset["topic_id"] == sel]["ts_hour"].unique()
        tv = videos[(videos["topic_id"] == sel) & (videos["ts_hour"].isin(topic_ts))].copy()

        # Platform mix
        if "platform" in tv.columns and not tv.empty:
            plat_mix = (tv.groupby("platform", as_index=False)
                          .agg(momentum=("decayed_velocity","sum"),
                               posts=("videoId","nunique"))
                          .sort_values("momentum", ascending=False))
            st.write("**Platform mix**")
            st.dataframe(plat_mix, use_container_width=True, hide_index=True)

        # Clickable YouTube link column
        def mk_link(row):
            if "platform" in row and str(row["platform"]).lower() != "youtube":
                return None
            vid = str(row["videoId"])
            return f"https://www.youtube.com/watch?v={vid}" if len(vid) == 11 else None
        tv["url"] = tv.apply(mk_link, axis=1)

        # Channel contributions
        contrib = (tv.groupby("channelTitle", as_index=False)
                     .agg(contrib=("decayed_velocity","sum"))
                     .sort_values("contrib", ascending=False))
        if not contrib.empty:
            fig3 = px.bar(contrib.head(10), x="contrib", y="channelTitle",
                          orientation="h", title="Top contributing channels")
            fig3.update_layout(yaxis={'categoryorder':'total ascending'},
                               height=400, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig3, use_container_width=True)

        # Search within titles
        q = st.text_input("Filter titles (search)", "")
        t_videos = tv.sort_values("decayed_velocity", ascending=False)
        if q:
            q_lower = q.lower()
            t_videos = t_videos[t_videos["title_clean"].str.lower().str.contains(q_lower, na=False)]

        # Columns for examples table
        cols = ["ts_hour_local","channelTitle","title_clean","lang","decayed_velocity","url"]
        if "platform" in t_videos.columns:
            cols = ["ts_hour_local","platform","channelTitle","title_clean","lang","decayed_velocity","url"]

        st.write(f"**Examples ({len(t_videos)})**")
        st.dataframe(
            t_videos[cols].rename(columns={"ts_hour_local":"time (Colombo)","url":"link"}),
            use_container_width=True, hide_index=True
        )

        # Download examples CSV
        vbuf = io.StringIO(); t_videos[cols].to_csv(vbuf, index=False)
        st.download_button("Download topic videos (CSV)", vbuf.getvalue(), "topic_videos.csv", "text/csv")

# === Channel Leaderboard ===
with tab_leader:
    st.subheader("Channel Leaderboard (momentum in window)")
    topic_hours = subset["ts_hour"].unique()
    vv = videos[videos["ts_hour"].isin(topic_hours)]
    board = (vv.groupby("channelTitle", as_index=False)
               .agg(momentum=("decayed_velocity","sum"),
                    posts=("videoId","nunique"))
               .sort_values("momentum", ascending=False)
               .head(25))
    st.dataframe(board, use_container_width=True, hide_index=True)

# -------- Footer --------
last_utc = pd.to_datetime(topics["ts_hour"].max()).tz_convert("UTC")
last_lk = last_utc.tz_convert("Asia/Colombo")
st.caption(f"Last updated (UTC): {last_utc:%Y-%m-%d %H:%M} | (Colombo): {last_lk:%Y-%m-%d %H:%M}")
