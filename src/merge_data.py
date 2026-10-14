import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lpad, trim

# Cấu hình PySpark trên Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

# Khởi tạo Spark
spark = (
    SparkSession.builder
    .appName("THPT_2026_Merge_Data")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.python.worker.reuse", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Đọc tất cả 34 file CSV
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)
    .option("encoding", "UTF-8")
    .csv("data/raw/*.csv")
)

# Chuẩn hóa mã số báo danh và mã tỉnh
df = (
    df.withColumn("sbd", lpad(trim(col("sbd")), 8, "0"))
      .withColumn(
          "province_code",
          lpad(trim(col("province_code")), 2, "0")
      )
)

# Kiểm tra số bản ghi
print("Tong so ban ghi:", df.count())

# Kiểm tra số tỉnh
print(
    "Tong so tinh/thanh:",
    df.select("province_code").distinct().count()
)

# Kiểm tra số báo danh bị trùng
duplicate_count = (
    df.groupBy("sbd")
      .count()
      .filter(col("count") > 1)
      .count()
)

print("So SBD bi trung:", duplicate_count)

# Hiển thị 10 bản ghi
df.show(10, truncate=False)

# Lưu dữ liệu thành Parquet
df.write.mode("overwrite").parquet(
    "data/processed/thpt_2026.parquet"
)

print("Da luu du lieu Parquet thanh cong!")

spark.stop()