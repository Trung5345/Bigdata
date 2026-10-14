
import os
import sys
import time

import pandas as pd

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

if os.path.isdir(r"C:\hadoop\bin"):
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] = (
        r"C:\hadoop\bin"
        + os.pathsep
        + os.environ.get("PATH", "")
    )

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

INPUT = "data/processed/thpt_2026_clean.parquet"
OUTPUT = "data/analysis/benchmark_results.csv"

os.makedirs("data/analysis", exist_ok=True)

results = []

# ==========================================
# 1. PANDAS
# ==========================================

print("\n=== THUC NGHIEM PANDAS ===")

start = time.perf_counter()

pdf = pd.read_parquet(
    INPUT,
    columns=["province_code", "toan"]
)

pdf["toan"] = pd.to_numeric(
    pdf["toan"],
    errors="coerce"
)

pdf = pdf[
    pdf["toan"].between(0, 10)
]

pandas_result = (
    pdf.groupby("province_code")["toan"]
    .agg(["count", "mean"])
    .reset_index()
    .sort_values("province_code")
)

pandas_time = time.perf_counter() - start

print("Thoi gian:", round(pandas_time, 3), "giay")
print("So tinh:", len(pandas_result))

results.append({
    "technology": "Pandas",
    "seconds": pandas_time,
    "province_count": len(pandas_result)
})

# ==========================================
# 2. PYSPARK
# ==========================================

print("\n=== THUC NGHIEM PYSPARK ===")

spark = (
    SparkSession.builder
    .appName("THPT_2026_Benchmark")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    start = time.perf_counter()

    sdf = spark.read.parquet(INPUT).select(
        "province_code", "toan"
    )

    spark_result = (
        sdf.filter(
            F.col("toan").between(0, 10)
        )
        .groupBy("province_code")
        .agg(
            F.count("*").alias("count"),
            F.avg("toan").alias("mean")
        )
        .orderBy("province_code")
        .collect()
    )

    spark_time = time.perf_counter() - start

    print(
        "Thoi gian:",
        round(spark_time, 3),
        "giay"
    )
    print("So tinh:", len(spark_result))

    results.append({
        "technology": "PySpark",
        "seconds": spark_time,
        "province_count": len(spark_result)
    })

    # ======================================
    # 3. KIEM TRA KET QUA
    # ======================================

    spark_check = pd.DataFrame(
        [r.asDict() for r in spark_result]
    )

    pandas_check = pandas_result.copy()

    spark_check["province_code"] = (
        spark_check["province_code"]
        .astype(str).str.zfill(2)
    )

    pandas_check["province_code"] = (
        pandas_check["province_code"]
        .astype(str).str.zfill(2)
    )

    comparison = pandas_check.merge(
        spark_check,
        on="province_code",
        suffixes=("_pandas", "_spark"),
        validate="one_to_one"
    )

    counts_match = (
        comparison["count_pandas"]
        == comparison["count_spark"]
    ).all()

    max_mean_diff = (
        comparison["mean_pandas"]
        - comparison["mean_spark"]
    ).abs().max()

    print("\n=== DOI CHIEU KET QUA ===")
    print("So tinh doi chieu:", len(comparison))
    print("So bai khop:", counts_match)
    print(
        "Sai lech diem TB lon nhat:",
        max_mean_diff
    )

finally:
    spark.stop()

# ==========================================
# 4. LUU KET QUA
# ==========================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n=== TONG HOP ===")
print(result_df.to_string(index=False))

print("\nDa luu:", OUTPUT)
