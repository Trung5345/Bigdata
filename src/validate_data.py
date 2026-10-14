import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum, when

spark = (
    SparkSession.builder
    .appName("THPT_2026_Validate")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Đọc dữ liệu đã làm sạch
df = spark.read.parquet(
    "data/processed/thpt_2026_clean.parquet"
)

subjects = [
    "toan", "ngu_van", "ngoai_ngu",
    "vat_li", "hoa_hoc", "sinh_hoc",
    "lich_su", "dia_li", "gdcd",
    "tin_hoc", "cong_nghe"
]

# Kiểm tra tổng quan
print("Tong so thi sinh:", df.count())

print(
    "So tinh/thanh:",
    df.select("province_code").distinct().count()
)

# Thống kê số điểm hợp lệ và không hợp lệ từng môn
for subject in subjects:
    result = df.agg(
        count(col(subject)).alias("co_diem"),
        spark_sum(
            when(
                col(subject).isNotNull() &
                ((col(subject) < 0) | (col(subject) > 10)),
                1
            ).otherwise(0)
        ).alias("diem_ngoai_khoang")
    ).first()

    print(
        f"{subject}: "
        f"{result['co_diem']} bai co diem, "
        f"{result['diem_ngoai_khoang']} diem ngoai khoang"
    )

# Kiểm tra dữ liệu Toán theo tỉnh
print("\nSo bai thi Toan theo tinh:")
df.groupBy("province_name").agg(
    count("toan").alias("so_bai_toan")
).orderBy(
    col("so_bai_toan").desc()
).show(34, truncate=False)

spark.stop()