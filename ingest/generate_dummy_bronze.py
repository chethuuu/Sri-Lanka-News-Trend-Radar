# ingest/generate_dummy_bronze.py
import os, json, random, argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

BRONZE = Path("data/bronze")
CONFIG = Path("config/channels.csv")
BRONZE.mkdir(parents=True, exist_ok=True)

DEFAULT_CHANNELS = [
    ("Ada Derana",             "UCCK3OZi788Ok44K97WAhLKQ"),
    ("Hiru News",              "UCckltLEhFLv8Xz_lQhYfwmg"),
    ("Newsfirst Sri Lanka",    "UCgnFSj7jQffD5V5m05j4dPw"),
    ("Sirasa TV",              "UCn0XmAUFv6d2tofMFEesSNw"),
]

# A larger topic bank (mix EN/SI/TA hints)
TOPIC_SNIPPETS = [
    "Fuel price revision announced by CPC",
    "Heavy rains & flood alerts in Western Province",
    "Sri Lanka vs India — match highlights",
    "Parliament debate on budget proposals",
    "New expressway opening ceremony",
    "Public sector salary adjustment discussion",
    "Electricity tariff revision rumours",
    "University admissions & UGC notice",
    "Colombo traffic update and diversions",
    "Foreign reserves & CBSL monthly report",
    "Dengue control program intensified",
    "Railway strike impacts commuters",
    "Power outage schedule update",
    "Tea auction prices rise in Colombo",
    "Rupee strengthens against major currencies",
    "Tourism arrivals improve in August",
    "Harbour expansion project milestone",
    "Gas price revision expected",
    "School term timetable update",
    "Weather advisory for coastal areas",
]

LANG_TAILS = ["", " | සිංහල", " | தமிழ்", " | English"]
LANG_WEIGHTS = [0.45, 0.35, 0.10, 0.10]  # tweak mix

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def load_channels_from_config():
    if not CONFIG.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(CONFIG)
        if "channel_id" in df.columns:
            chans = []
            for _, r in df.iterrows():
                name = str(r.get("name") or r["channel_id"])
                cid = str(r["channel_id"]).strip()
                chans.append((name, cid))
            return chans
    except Exception:
        pass
    return None

def synthesize_channels(n: int, seed=42):
    random.seed(seed)
    chans = []
    for i in range(n):
        name = f"Channel {i+1:03d}"
        # Fake but UC-like ID
        cid = "UC" + "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789") for _ in range(22))
        chans.append((name, cid))
    return chans

def generate_records(channels, videos_per_channel, hours_back, seed=42, trend_bursts=4):
    """
    Simulate uploads over the last `hours_back` hours.
    Add `trend_bursts` global topics that many channels post about around the same time.
    """
    random.seed(seed)
    now = datetime.now(timezone.utc)

    # Pick some burst topics + when they spike
    burst_topics = random.sample(TOPIC_SNIPPETS, k=min(trend_bursts, len(TOPIC_SNIPPETS)))
    burst_times = [now - timedelta(hours=random.uniform(2, hours_back-2)) for _ in burst_topics]

    records = []
    for ch_title, ch_id in channels:
        # channel personality
        ch_base_rate = random.uniform(20, 180)  # views/min baseline
        ch_var = random.uniform(0.5, 1.6)

        for i in range(videos_per_channel):
            # Random publish time in the window
            pub_dt = now - timedelta(hours=random.uniform(0.1, hours_back))
            fetched_at = now

            # Some videos align with a burst (higher views, more channels involved)
            title_base = random.choice(TOPIC_SNIPPETS)
            for (bt, bt_time) in zip(burst_topics, burst_times):
                if abs((pub_dt - bt_time).total_seconds()) < 90*60:  # within 90 minutes of burst time
                    if random.random() < 0.35:  # 35% of videos around this time use burst topic
                        title_base = bt
                    break

            lang_tail = random.choices(LANG_TAILS, weights=LANG_WEIGHTS, k=1)[0]
            title = f"{title_base}{lang_tail}"

            # Unique video id
            video_id = f"vid_{ch_title.replace(' ', '')}_{i}_{int(pub_dt.timestamp())}_{random.randint(1000,9999)}"

            age_min = max((fetched_at - pub_dt).total_seconds() / 60.0, 1.0)

            # Simulate views: channel base * age * noise; bursts get a boost
            burst_boost = 1.0
            if title_base in burst_topics:
                burst_boost = random.uniform(1.5, 3.5)
            views = int(ch_base_rate * age_min * ch_var * burst_boost * random.uniform(0.7, 1.3))
            likes = int(max(0, views * random.uniform(0.006, 0.03)))
            comments = int(max(0, views * random.uniform(0.001, 0.01)))

            rec = {
                "id": video_id,  # our cleaner accepts string id as videoId
                "snippet": {
                    "title": title,
                    "description": f"{title} - full bulletin and details inside.",
                    "channelId": ch_id,
                    "channelTitle": ch_title,
                    "publishedAt": iso(pub_dt),
                },
                "statistics": {
                    "viewCount": views,
                    "likeCount": likes,
                    "commentCount": comments,
                },
                "_fetched_at": iso(fetched_at),
            }
            records.append(rec)
    random.shuffle(records)
    return records

def write_jsonl(records):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = BRONZE / f"youtube_raw_{ts}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Generate large dummy Bronze dataset.")
    ap.add_argument("--videos-per-channel", type=int, default=80, help="How many videos per channel")
    ap.add_argument("--num-channels", type=int, default=150, help="Number of channels to synthesize if config not present")
    ap.add_argument("--hours-back", type=float, default=72.0, help="Time window to simulate (hours)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trend-bursts", type=int, default=5, help="Number of global trending bursts")
    args = ap.parse_args()

    random.seed(args.seed)

    # Prefer config/channels.csv if present; else seed with defaults + synthetic to reach target
    channels = load_channels_from_config()
    if channels is None:
        channels = list(DEFAULT_CHANNELS)
        args.num_channels = args.num_channels - len(channels)
        if args.num_channels > 0:
            channels += synthesize_channels(args.num_channels, seed=args.seed)

    # Generate
    records = generate_records(
        channels=channels,
        videos_per_channel=args.videos_per_channel,
        hours_back=args.hours_back,
        seed=args.seed,
        trend_bursts=args.trend_bursts,
    )
    out = write_jsonl(records)
    print(f"Dummy Bronze written: {out}  | records: {len(records)}")

if __name__ == "__main__":
    main()
