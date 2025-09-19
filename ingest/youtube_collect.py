import os, json, time, argparse
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from googleapiclient.discovery import build
import pandas as pd
from pathlib import Path

BRONZE = Path("data/bronze"); BRONZE.mkdir(parents=True, exist_ok=True)
STATE  = Path("data/seen_ids.json"); STATE.parent.mkdir(parents=True, exist_ok=True)

def load_seen():
    if STATE.exists():
        try: return set(json.load(open(STATE)))
        except: return set()
    return set()

def save_seen(seen: set):
    json.dump(sorted(seen), open(STATE, "w"))

def bronze_dump(records, prefix="youtube_raw"):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = BRONZE / f"{prefix}_{ts}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(path)

def get_uploads_playlist_id(yt, channel_id: str) -> str:
    resp = yt.channels().list(part="contentDetails", id=channel_id).execute()
    items = resp.get("items", [])
    if not items: raise ValueError(f"No channel found: {channel_id}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

def list_videos_from_uploads(yt, uploads_pid: str, published_after_iso: str, per_channel_cap: int):
    vids = []
    token = None
    cutoff = datetime.fromisoformat(published_after_iso.replace("Z","")).replace(tzinfo=timezone.utc) if published_after_iso else None
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
            ts  = cd.get("videoPublishedAt")
            if cutoff and ts:
                ts_dt = datetime.fromisoformat(ts.replace("Z","")).replace(tzinfo=timezone.utc)
                if ts_dt < cutoff:
                    # playlist is in reverse chrono; once older than cutoff, we can stop
                    return vids
            vids.append(vid)
            if len(vids) >= per_channel_cap:
                return vids
        token = resp.get("nextPageToken")
        if not token: break
    return vids

def fetch_video_stats(yt, video_ids):
    out = []
    for i in range(0, len(video_ids), 50):
        resp = yt.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids[i:i+50])
        ).execute()
        out.extend(resp.get("items", []))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2, help="Look back window in days (recent only = fewer quota)")
    ap.add_argument("--cap", type=int, default=120, help="Max videos per channel this run")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between channels (sec)")
    ap.add_argument("--channels_csv", type=str, default="config/channels.csv")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    assert api_key, "Set YOUTUBE_API_KEY in .env"
    yt = build("youtube", "v3", developerKey=api_key)

    channels = pd.read_csv(args.channels_csv)
    assert "channel_id" in channels.columns, "channels.csv must have 'channel_id'"

    published_after = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()

    seen = load_seen()
    all_items = []

    for _, row in channels.iterrows():
        ch_name = str(row.get("name") or "").strip()
        ch_id   = str(row["channel_id"]).strip()
        try:
            uploads = get_uploads_playlist_id(yt, ch_id)
            ids = list_videos_from_uploads(yt, uploads, published_after, args.cap)
            ids = [vid for vid in ids if vid not in seen]
            if not ids:
                print(f"[INFO] {ch_name or ch_id}: no new videos in window")
                continue
            items = fetch_video_stats(yt, ids)
            now_iso = datetime.now(timezone.utc).isoformat()
            for it in items:
                it["_fetched_at"] = now_iso
            all_items.extend(items)
            seen.update([it["id"] for it in items if "id" in it])
            save_seen(seen)
            time.sleep(args.sleep)
        except Exception as e:
            print(f"[WARN] {ch_name or ch_id}: {e}")

    if not all_items:
        print("No items fetched.")
        return
    out = bronze_dump(all_items, "youtube_raw")
    print("Bronze dump written:", out, "| records:", len(all_items))

if __name__ == "__main__":
    main()
