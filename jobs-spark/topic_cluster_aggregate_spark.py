# Silver -> Gold with PySpark ML
# TF-IDF + KMeans topics, hourly aggregates, rising score, leader channels.

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, NGram
from pyspark.ml.clustering import KMeans
from pyspark.sql.window import Window
from pathlib import Path

SILVER = "data/silver/youtube_clean.parquet"
GOLD   = "data/gold"

spark = (
    SparkSession.builder
    .appName("topic_cluster_aggregate_spark")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
Path(GOLD).mkdir(parents=True, exist_ok=True)

def main():
    df = spark.read.parquet(SILVER)
    if df.rdd.isEmpty():
        print("No silver data.")
        return

    # Tokenize titles with unicode-safe pattern (letters, marks, digits)
    tokenizer = RegexTokenizer(
        inputCol="title_clean", outputCol="tokens",
        pattern=r"[^\p{L}\p{M}\p{N}]+",  # split on non-letter/mark/number
        gaps=True, toLowercase=True
    )

    # Light stopword removal for English only (Sinhala/Tamil left as-is)
    remover = StopWordsRemover(inputCol="tokens", outputCol="tokens_sw")

    # Add bigrams + unigrams (concat later)
    ngram = NGram(n=2, inputCol="tokens_sw", outputCol="bigrams")

    # HashingTF (bag-of-words) + IDF
    tf = HashingTF(inputCol="tokens_sw", outputCol="tf_raw", numFeatures=1_000_000)
    idf = IDF(inputCol="tf_raw", outputCol="features")

    pipe = Pipeline(stages=[tokenizer, remover, ngram, tf, idf])
    model = pipe.fit(df.fillna({"title_clean": ""}))
    fe = model.transform(df.fillna({"title_clean": ""}))

    # Choose k adaptively (similar heuristic as pandas version)
    n_docs = fe.count()
    if n_docs >= 40:
        k = min(60, max(4, n_docs // 80))
    else:
        k = max(2, n_docs // 5) or 2

    kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="topic_id")
    km_model = kmeans.fit(fe)
    fe = km_model.transform(fe)

    # Hour bucket UTC
    fe = fe.withColumn("ts_hour", F.date_trunc("hour", F.col("fetchedAt")))

    # Leader channels per topic/hour
    leaders = (
        fe.groupBy("ts_hour", "topic_id", "channelTitle")
          .agg(F.sum("decayed_velocity").alias("weight"))
    )
    w_rank = Window.partitionBy("ts_hour", "topic_id").orderBy(F.col("weight").desc())
    leaders = leaders.withColumn("rank", F.row_number().over(w_rank))
    top3 = (
        leaders.where(F.col("rank") <= 3)
               .groupBy("ts_hour", "topic_id")
               .agg(F.concat_ws(", ", F.collect_list("channelTitle")).alias("leader_channels"))
    )

    # Hourly topic aggregates
    agg = (
        fe.groupBy("ts_hour", "topic_id")
          .agg(
              F.sum("decayed_velocity").alias("velocity_sum"),
              F.countDistinct("videoId").alias("video_count"),
              F.countDistinct("channelId").alias("channel_count"),
              F.last("title_clean", ignorenulls=True).alias("sample_title")
          )
          .join(top3, on=["ts_hour", "topic_id"], how="left")
    )

    # Rising score vs expanding baseline (per topic)
    w_time = Window.partitionBy("topic_id").orderBy("ts_hour").rowsBetween(Window.unboundedPreceding, -1)
    vel_mean = F.avg("velocity_sum").over(w_time)
    vel_std  = F.stddev("velocity_sum").over(w_time)

    agg = (agg
           .withColumn("vel_mean", vel_mean)
           .withColumn("vel_std",  F.when(F.col("vel_std").isNull(), F.lit(1.0)).otherwise(vel_std))
           .fillna({"vel_mean": 0.0, "vel_std": 1.0})
           .withColumn("zscore", (F.col("velocity_sum") - F.col("vel_mean")) / F.when(F.col("vel_std")==0, 1.0).otherwise(F.col("vel_std")))
           .withColumn("rising_flag", (F.col("zscore") >= F.lit(2.0)).cast("int"))
           .withColumn("rising_score", (F.when(F.col("zscore") < 0, 0).otherwise(F.least(F.col("zscore"), F.lit(5))) / 5.0 * 100.0))
    )

    # Write GOLD (as CSV dirs for big data friendliness)
    (agg
     .select("ts_hour","topic_id","leader_channels","velocity_sum","video_count","channel_count","zscore","rising_flag","rising_score","sample_title")
     .coalesce(1)
     .write.mode("overwrite").option("header", True).csv(f"{GOLD}/topics_hourly.csv_dir")
    )

    (fe
     .select("videoId","channelId","channelTitle","title_clean","lang","decayed_velocity","ts_hour","topic_id","platform")
     .coalesce(1)
     .write.mode("overwrite").option("header", True).csv(f"{GOLD}/videos_with_topics.csv_dir")
    )

    print(f"Wrote GOLD folders: {GOLD}/topics_hourly.csv_dir and {GOLD}/videos_with_topics.csv_dir | k={k} | docs={n_docs}")

if __name__ == "__main__":
    main()
