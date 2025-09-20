# jobs/clean_and_features_spark.py
# Bronze -> Silver using PySpark
# Reads JSONL from data/bronze, flattens YouTube payloads, computes velocity features,
# writes Parquet (and a small CSV dir for checks).

from pyspark.sql import SparkSession, functions as F, types as T
from pathlib import Path
import math

BRONZE = "data/bronze"
SILVER = "data/silver"

spark = (
    SparkSession.builder
    .appName("clean_and_features_spark")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
Path(SILVER).mkdir(parents=True, exist_ok=True)

def main():
    # Read all bronze JSONL (skip if none)
    bronze = spark.read.json(f"{BRONZE}/youtube_raw_*.jsonl", multiLine=False)
    if bronze.rdd.isEmpty():
        # create empty Silver parquet to keep downstream happy
        empty = spark.createDataFrame([], schema=T.StructType([
            T.StructField("platform", T.StringType()),
            T.StructField("videoId", T.StringType())
        ]))
        empty.write.mode("overwrite").parquet(f"{SILVER}/youtube_clean.parquet")
        print("No bronze files found; wrote empty silver parquet.")
        return

    # Default platform = youtube
    platform_col = F.coalesce(F.col("platform"), F.lit("youtube"))

    # Helper getters
    def get_video_id(col):
        # id is either string or struct with videoId
        return F.when(F.col("id").cast("string").isNotNull() & (F.col("id").dtype == "string"), F.col("id")) \
                .otherwise(F.col("id.videoId"))

    # Title: prefer snippet.title, else snippet.localized.title
    title_col = F.coalesce(F.col("snippet.title"), F.col("snippet.localized.title"))

    # Duration (optional): contentDetails.duration like "PT1M30S"
    # Parse to seconds in Spark (small expr)
    @F.udf(T.IntegerType())
    def iso8601_to_seconds(s):
        if not s or not isinstance(s, str):
            return None
        # Simple parse for PT#H#M#S, optional days 'P#DT'
        import re
        m = re.match(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$", s)
        if not m:
            return None
        days = int(m.group(1) or 0)
        hours = int(m.group(2) or 0)
        mins  = int(m.group(3) or 0)
        secs  = int(m.group(4) or 0)
        return days*86400 + hours*3600 + mins*60 + secs

    df = bronze.select(
        platform_col.alias("platform"),
        get_video_id("id").alias("videoId"),
        title_col.alias("title"),
        F.col("snippet.description").alias("description"),
        F.col("snippet.channelId").alias("channelId"),
        F.col("snippet.channelTitle").alias("channelTitle"),
        F.to_timestamp(F.col("snippet.publishedAt")).alias("publishedAt"),
        F.to_timestamp(F.col("_fetched_at")).alias("fetchedAt"),
        F.col("statistics.viewCount").cast("double").alias("views"),
        F.col("statistics.likeCount").cast("double").alias("likes"),
        F.col("statistics.commentCount").cast("double").alias("comments"),
        iso8601_to_seconds(F.col("contentDetails.duration")).alias("duration_sec")
    )

    # Drop missing keys
    df = df.dropna(subset=["videoId", "publishedAt", "fetchedAt"])

    # Cap future publishes (rare clock skew): publishedAt > fetchedAt => set to fetchedAt
    df = df.withColumn(
        "publishedAt",
        F.when(F.col("publishedAt") > F.col("fetchedAt"), F.col("fetchedAt")).otherwise(F.col("publishedAt"))
    )

    # Deduplicate latest snapshot per (platform, videoId)
    w = F.window("fetchedAt", "36500 days")  # dummy for ordering; we’ll use row_number over partition
    from pyspark.sql.window import Window
    win = Window.partitionBy("platform", "videoId").orderBy(F.col("fetchedAt").asc())
    df = (
        df.withColumn("rn", F.row_number().over(win))
          .withColumn("maxrn", F.max("rn").over(Window.partitionBy("platform", "videoId")))
          .where(F.col("rn") == F.col("maxrn"))
          .drop("rn", "maxrn")
    )

    # Clean title; (language detection in Spark is heavy — skip here or do in pandas later)
    df = df.withColumn("title_clean", F.trim(F.regexp_replace(F.coalesce(F.col("title"), F.lit("")), r"\s+", " ")))

    # Fill metric nulls
    df = df.fillna({"views": 0.0, "likes": 0.0, "comments": 0.0})

    # Age (minutes) & per-minute rates
    df = df.withColumn("age_min", F.greatest(F.lit(0.0001),
                    (F.col("fetchedAt").cast("long") - F.col("publishedAt").cast("long")) / F.lit(60.0)))

    df = df.withColumn("views_per_min",    F.col("views")    / F.col("age_min")) \
           .withColumn("likes_per_min",    F.col("likes")    / F.col("age_min")) \
           .withColumn("comments_per_min", F.col("comments") / F.col("age_min"))

    # Clip outliers at ~99.9th percentile (per column)
    def clip99p(colname):
        q = df.approxQuantile(colname, [0.999], 0.01)[0]
        return F.when(F.col(colname) > F.lit(q), F.lit(q)).otherwise(F.col(colname)) if q is not None else F.col(colname)

    df = df.withColumn("views_per_min", clip99p("views_per_min")) \
           .withColumn("likes_per_min", clip99p("likes_per_min")) \
           .withColumn("comments_per_min", clip99p("comments_per_min"))

    # Recency-decayed momentum (tau=120min)
    tau = 120.0
    @F.udf(T.DoubleType())
    def decay(vpm, age):
        try:
            return float(vpm) * math.e ** (-(float(age) / tau))
        except Exception:
            return 0.0

    df = df.withColumn("decayed_velocity", decay(F.col("views_per_min"), F.col("age_min")))

    # Write Silver
    out_parq = f"{SILVER}/youtube_clean.parquet"
    out_csvd = f"{SILVER}/youtube_clean.csv_dir"
    df.write.mode("overwrite").parquet(out_parq)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_csvd)

    print(f"Wrote: {out_parq} and {out_csvd}")

if __name__ == "__main__":
    main()
