import os
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from pytrends.request import TrendReq

BRONZE = Path("data/bronze")
BRONZE.mkdir(parents=True, exist_ok=True)


def bronze_dump(records, prefix):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = BRONZE / f"{prefix}_{ts}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="+", required=True,
                    help="Queries/keywords to track (space-separated list)")
    ap.add_argument("--days", type=int, default=7,
                    help="Lookback (<=7 gives hourly resolution in Google Trends)")
    ap.add_argument("--geo", default="LK", help="Geography (LK = Sri Lanka)")
    args = ap.parse_args()

    # Hourly data is available up to 7 days: timeframe like 'now 7-d'
    timeframe = f"now {min(args.days, 7)}-d"

    pytrends = TrendReq(hl="en-US", tz=0)
    pytrends.build_payload(kw_list=args.queries,
                           timeframe=timeframe, geo=args.geo)

    # index = datetime (UTC-ish), columns = queries, values 0-100
    df = pytrends.interest_over_time()
    if df.empty:
        print("No Trends data returned.")
        return

    fetched_at = datetime.now(timezone.utc).isoformat()
    records = []

    # Emit Google-Trends rows in the SAME bronze schema your cleaner expects
    for ts, row in df.drop(columns=["isPartial"]).iterrows():
        publishedAt = ts.tz_localize("UTC").isoformat(
        ) if ts.tzinfo is None else ts.astimezone(timezone.utc).isoformat()
        for q in row.index:
            interest = int(row[q]) if pd.notna(row[q]) else 0
            rec = {
                "platform": "trends",                     # <— NEW
                "id": f"trends_{q}_{int(ts.timestamp())}",  # unique id
                "snippet": {
                    "title": q,                           # treat the query like a "title"
                    "description": f"Google Trends interest for '{q}'",
                    "channelId": "trends:LK",
                    "channelTitle": "Google Trends (LK)",
                    "publishedAt": publishedAt,           # hour bucket time
                },
                "statistics": {
                    "viewCount": interest,                # map interest → views
                    "likeCount": 0,
                    "commentCount": 0,
                },
                "_fetched_at": fetched_at,
            }
            records.append(rec)

    out = bronze_dump(records, "trends_raw")
    print(f"Bronze (Trends) written: {out} | records: {len(records)}")


if __name__ == "__main__":
    main()
