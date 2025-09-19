# ingest/generate_dummy_bronze.py
import os, json, random
from pathlib import Path
from datetime import datetime, timedelta, timezone

random.seed(42)

BRONZE = Path("data/bronze")
BRONZE.mkdir(parents=True, exist_ok=True)

# Some SL news channels (fake IDs/titles are fine for dummy data)
CHANNELS = [
    ("Ada Derana",             "UCCK3OZi788Ok44K97WAhLKQ"),
    ("Hiru News",              "UCckltLEhFLv8Xz_lQhYfwmg"),
    ("Newsfirst Sri Lanka",    "UCgnFSj7jQffD5V5m05j4dPw"),
    ("Sirasa TV",              "UCn0XmAUFv6d2tofMFEesSNw"),
]

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
]

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def main():
    now = datetime.now(timezone.utc)
    records = []

    # Generate ~30 videos per channel over the last 48 hours (total ~120)
    for ch_title, ch_id in CHANNELS:
        for i in range(30):
            # publish time between now-48h and now-10min
            pub_dt = now - timedelta(hours=random.uniform(0.2, 48.0))
            # fetched at ~ now (simulate you ran the collector now)
            fetched_at = now

            # Make IDs unique-ish
            video_id = f"vid_{ch_title.replace(' ', '')}_{i}_{int(pub_dt.timestamp())}"

            # Title picks + a bit of variation
            base = random.choice(TOPIC_SNIPPETS)
            lang_tail = random.choice(["", " | සිංහල", " | தமிழ்", " | English"])
            title = f"{base}{lang_tail}"

            age_min = max((fetched_at - pub_dt).total_seconds() / 60.0, 1.0)
            # Views roughly follow a decaying growth; add random noise by channel
            base_rate = random.uniform(10, 200)  # views/min baseline
            views = int(base_rate * age_min * random.uniform(0.6, 1.6))
            likes = int(views * random.uniform(0.005, 0.03))
            comments = int(views * random.uniform(0.001, 0.01))

            rec = {
                "id": video_id,  # IMPORTANT: our cleaner supports string id ==> videoId
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

    # Write JSONL
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    out_path = BRONZE / f"youtube_raw_{ts}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Dummy Bronze written: {out_path}  | records: {len(records)}")

if __name__ == "__main__":
    main()
