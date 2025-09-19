import os
from pyspark.sql import SparkSession, functions as F, types as T
from sentence_transformers import SentenceTransformer
import numpy as np

SILVER = "data/silver"
spark = (SparkSession.builder.appName("embeddings").getOrCreate())

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@F.udf(T.ArrayType(T.FloatType()))
def embed(text):
    if text is None:
        return None
    vec = model.encode([text])[0].astype(np.float32)
    return [float(x) for x in vec]

def main():
    df = spark.read.parquet(f"{SILVER}/youtube_clean.parquet")
    df = df.withColumn("embed", embed(F.col("title_clean")))
    df.write.mode("overwrite").parquet(f"{SILVER}/youtube_with_embeds.parquet")
    print("Wrote silver/youtube_with_embeds.parquet")

if __name__ == "__main__":
    main()
