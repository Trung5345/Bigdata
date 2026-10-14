import os
import sys
import html
from collections import Counter

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = (
    SparkSession.builder
    .appName("THPT_Check_Subjects")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(
    "data/processed/thpt_2026_clean.parquet"
)

# Tách tên môn từ cột extra
def extract_subjects(extra):
    if not extra:
        return []

    decoded = html.unescape(extra)
    subjects = []

    for item in decoded.split(";"):
        if "=" in item:
            name = item.split("=", 1)[0].strip()
            subjects.append(name)

    return subjects

# Thống kê tên môn trên Spark, không thu thập toàn bộ
# 1.2 triệu dòng về Python driver
names = (
    df.select("extra")
    .where(col("extra").isNotNull())
    .rdd.flatMap(lambda row: extract_subjects(row["extra"]))
    .map(lambda name: (name, 1))
    .reduceByKey(lambda a, b: a + b)
    .collect()
)

print("\nDANH SACH MON TRONG EXTRA:")
for name, count in sorted(names, key=lambda x: -x[1]):
    print(f"{name}: {count}")

spark.stop()