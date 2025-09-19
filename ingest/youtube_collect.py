# ingest/youtube_collect_rss.py
import os, json, time
from datetime import datetime, timezone
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
from yt_dlp import YoutubeDL
import pandas as pd

BRONZE_DIR = Path("data/bronze")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

def bronze_dump(records, fname_prefix):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = BRONZE_DIR / f"{fname_prefix}_{ts}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(path)

def list_recent_videos_via_rss(channel_id: str, limit: int = 20):
    """
    Use YouTube channel RSS (no quota) to list recent video IDs + published times.
    """
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.text)

    # RSS uses the Atom namespace
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "yt": "http://www.youtube.com/xml/schemas/2015"
    }
    items = []
    for entry in root.findall("atom:entry", ns):
        vid_tag = entry.find("yt:videoId", ns)
        pub_tag = entry.find("atom:published", ns)
        if vid_tag is None or pub_tag is None:
            continue
        items.append({
            "videoId": vid_tag.text,
            "publishedAt": pub_tag.text
        })
    # newest first in feed; keep first N
    return [it["videoId"] for it in items[:limit]]

def fetch_video_details_with_ytdlp(video_ids):
    """
    Use yt-dlp to fetch metadata/stats without the API.
    We'll emit a schema similar to the API output so the rest of your pipeline works.
    """
    out = []
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "forcejson": True,
        "noplaylist": True,
        "consoletitle": False,
    }
    now_iso = datetime.now(timezone.utc).isoformat()

    with YoutubeDL(ydl_opts) as ydl:
        for vid in video_ids:
            url = f"https://www.youtube.com/watch?v={vid}"
            try:
                info = ydl.extract_info(url, download=False)
                # Map yt-dlp fields to our expected structure
                snippet = {
                    "title": info.get("title"),
                    "description": info.get("description"),
                    "channelId": info.get("channel_id"),
                    "channelTitle": info.get("channel"),
                    # Prefer upload_date if available, else timestamp
                    "publishedAt": None
                }
                if info.get("upload_date"):
                    # upload_date is 'YYYYMMDD'; keep as ISO date (no time)
                    d = info["upload_date"]
                    snippet["publishedAt"] = f"{d[0:4]}-{d[4:6]}-{d[6:8]}T00:00:00Z"
                elif info.get("timestamp"):
                    snippet["publishedAt"] = datetime.fromtimestamp(
                        info["timestamp"], tz=timezone.utc).isoformat()

                statistics = {
                    "viewCount": info.get("view_count"),
                    "likeCount": info.get("like_count"),  # may be None if hidden
                    "commentCount": info.get("comment_count"),  # often None
                }

                record = {
                    "id": vid,                      # NOTE: keep as string videoId (our pandas cleaner now supports this)
                    "snippet": snippet,
                    "statistics": statistics,
                    "_fetched_at": now_iso,
                }
                out.append(record)
                time.sleep(0.1)
            except Exception as e:
                print(f"[WARN] Failed {url}: {e}")
                continue
    return out

def main():
    channels = pd.read_csv("config/channels.csv")
    # channels.csv must have 'channel_id' column
    if "channel_id" not in channels.columns:
        raise RuntimeError("config/channels.csv must have a 'channel_id' column")

    all_items = []
    for _, row in channels.iterrows():
        ch_name = row.get("name", "")
        ch_id = str(row["channel_id"]).strip()
        try:
            video_ids = list_recent_videos_via_rss(ch_id, limit=20)
            if not video_ids:
                print(f"[INFO] No videos via RSS for {ch_name or ch_id}")
                continue
            items = fetch_video_details_with_ytdlp(video_ids)
            all_items.extend(items)
        except Exception as e:
            print(f"[WARN] {ch_name or ch_id}: {e}")

    if not all_items:
        print("No items fetched.")
        return

    out = bronze_dump(all_items, "youtube_raw")
    print("Bronze dump written:", out)

if __name__ == "__main__":
    main()
