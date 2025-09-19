"""
YouTube → Bronze (JSONL) ingestor with low-quota pattern.

Design choices (explain in viva):
- Avoids expensive search.list. Uses:
  channels.list(part=contentDetails) → uploads playlist (1 unit)
  playlistItems.list(part=contentDetails) → video ids (1 unit/page)
  videos.list(part=snippet,statistics,contentDetails) → batched details (1 unit per 50 ids)
- Caches seen video ids to prevent re-fetching and save quota.
- Writes Bronze as JSON Lines (append-friendly, raw-preserving).
- Adds 'platform' field for multi-source pipelines (e.g., Trends + YouTube).
"""

import os
import json
import time
import argparse
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd

# -------- Paths (Medallion: Bronze/Silver/Gold — this writes Bronze) --------
BRONZE = Path("data/bronze")
BRONZE.mkdir(parents=True, exist_ok=True)
STATE = Path("data/seen_ids.json")
STATE.parent.mkdir(parents=True, exist_ok=True)
# simple run log (file, rows, channels, window, fetched_at)
MANIFEST = BRONZE / "bronze_manifest.csv"

# -------- Small utilities --------


def load_seen() -> set:
    """Load the 'seen video ids' cache to avoid re-fetching across runs."""
    if STATE.exists():
        try:
            return set(json.load(open(STATE)))
        except Exception:
            return set()
    return set()


def save_seen(seen: set) -> None:
    """Persist the cache."""
    json.dump(sorted(seen), open(STATE, "w"))


def bronze_dump(records: list, prefix="youtube_raw") -> str:
    """Write a Bronze JSONL file and return its path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = BRONZE / f"{prefix}_{ts}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(path)


def get_uploads_playlist_id(yt, channel_id: str) -> str:
    """
    Cheap way to enumerate a channel's uploads:
    - channels.list(part=contentDetails) gives 'relatedPlaylists.uploads'
    """
    resp = yt.channels().list(part="contentDetails", id=channel_id).execute()
    items = resp.get("items", [])
    if not items:
        raise ValueError(f"No channel found: {channel_id}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def list_videos_from_uploads(yt, uploads_pid: str, published_after_iso: str, per_channel_cap: int) -> list:
    """
    Walk the uploads playlist (reverse-chronological), stop when:
    - we pass the published_after cutoff, or
    - we hit per_channel_cap, or
    - there is no next page.
    Returns raw videoId strings.
    """
    vids, token = [], None
    cutoff = (
        datetime.fromisoformat(published_after_iso.replace("Z", ""))
        .replace(tzinfo=timezone.utc) if published_after_iso else None
    )
    while True:
        resp = yt.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_pid,
            maxResults=50,
            pageToken=token
        ).execute()

        for it in resp.get("items", []):
            cd = it["contentDetails"]
            vid = cd["videoId"]
            ts = cd.get("videoPublishedAt")  # ISO8601 Z
            if cutoff and ts:
                ts_dt = datetime.fromisoformat(ts.replace(
                    "Z", "")).replace(tzinfo=timezone.utc)
                # Because playlist is newest→oldest, we can early-stop once older than cutoff
                if ts_dt < cutoff:
                    return vids
            vids.append(vid)
            if len(vids) >= per_channel_cap:
                return vids

        token = resp.get("nextPageToken")
        if not token:
            break
    return vids


def fetch_video_stats(yt, video_ids: list) -> list:
    """
    Batch-fetch details for video ids (≤50 per call).
    Returns items in the native YouTube API 'videos.list' shape.
    """
    out = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        if not chunk:
            continue
        resp = yt.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(chunk)
        ).execute()
        out.extend(resp.get("items", []))
    return out

# -------- Main CLI entry --------


def main():
    ap = argparse.ArgumentParser(
        description="Ingest YouTube videos into Bronze JSONL (low-quota pattern).")
    ap.add_argument("--days", type=int, default=2,
                    help="Lookback window in days (smaller = fewer API calls)")
    ap.add_argument("--cap", type=int, default=120,
                    help="Max videos per channel for this run")
    ap.add_argument("--sleep", type=float, default=0.2,
                    help="Pause between channels (seconds)")
    ap.add_argument("--channels_csv", type=str, default="config/channels.csv",
                    help="CSV with columns: name,channel_id (UC...)")
    args = ap.parse_args()

    # Load API key (store in .env as YOUTUBE_API_KEY=...)
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    assert api_key, "Set YOUTUBE_API_KEY in .env"
    yt = build("youtube", "v3", developerKey=api_key)

    # Load channels (must be channel IDs, not @handles)
    channels = pd.read_csv(args.channels_csv)
    assert "channel_id" in channels.columns, "channels.csv must have a 'channel_id' column"

    # Compute lookback cutoff (ISO8601 with 'Z' for clarity)
    published_after = (datetime.now(timezone.utc) -
                       timedelta(days=args.days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_started_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    seen = load_seen()
    all_items = []

    # Iterate channels (keep polite delays; handle quota/rate with light backoff)
    for _, row in channels.iterrows():
        ch_name = str(row.get("name") or "").strip()
        ch_id = str(row["channel_id"]).strip()

        try:
            uploads = get_uploads_playlist_id(yt, ch_id)
            ids = list_videos_from_uploads(
                yt, uploads, published_after, args.cap)
            # Skip ids we've already processed in previous runs (cache)
            ids = [vid for vid in ids if vid not in seen]

            if not ids:
                print(f"[INFO] {ch_name or ch_id}: no new videos in window")
                time.sleep(args.sleep)
                continue

            # Fetch details in batches
            items = fetch_video_stats(yt, ids)

            # Stamp fetched time and platform (helps downstream UI)
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            for it in items:
                it["_fetched_at"] = now_iso
                it["platform"] = "youtube"

            all_items.extend(items)

            # Update cache and pause briefly
            seen.update([it["id"] for it in items if "id" in it])
            save_seen(seen)
            time.sleep(args.sleep)

        except HttpError as e:
            # Graceful handling of quota/rate; brief randomized sleep then continue
            if e.resp.status in (403, 429):
                wait = args.sleep + random.uniform(0.2, 0.8)
                print(
                    f"[WARN] {ch_name or ch_id}: quota/rate limit ({e.resp.status}). Sleeping {wait:.2f}s")
                time.sleep(wait)
                continue
            else:
                print(f"[WARN] {ch_name or ch_id}: {e}")
                continue
        except Exception as e:
            print(f"[WARN] {ch_name or ch_id}: {e}")
            continue

    if not all_items:
        print("No items fetched.")
        return

    # Write Bronze JSONL
    out = bronze_dump(all_items, "youtube_raw")
    print("Bronze dump written:", out, "| records:", len(all_items))

    # Append a manifest row (nice for the report/viva)
    try:
        write_header = not MANIFEST.exists() or MANIFEST.stat().st_size == 0
        with MANIFEST.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["file", "records", "channels",
                           "published_after", "fetched_at"])
            w.writerow([out, len(all_items), len(channels),
                       published_after, run_started_iso])
        print("Updated manifest:", MANIFEST)
    except Exception as e:
        print(f"[WARN] Could not write manifest: {e}")


if __name__ == "__main__":
    main()
