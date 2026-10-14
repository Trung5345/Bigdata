import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim

spark = (
    SparkSession.builder
    .appName("THPT_Check_Data")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet("data/processed/thpt_2026.parquet")

# Đếm bản ghi có dữ liệu trong cột extra
extra_count = df.filter(
    col("extra").isNotNull() &
    (trim(col("extra")) != "")
).count()

# Đếm bản ghi thiếu điểm Toán
missing_math = df.filter(
    col("toan").isNull() |
    (trim(col("toan")) == "")
).count()

print("Tong so ban ghi:", df.count())
print("So ban ghi co cot extra:", extra_count)
print("So ban ghi thieu diem Toan:", missing_math)

# Thống kê cột extra theo tỉnh
df.filter(
    col("extra").isNotNull() &
    (trim(col("extra")) != "")
).groupBy("province_name").count() \
 .orderBy(col("count").desc()) \
 .show(34, truncate=False)

spark.stop()