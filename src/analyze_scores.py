
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

if os.path.isdir(r"C:\hadoop\bin"):
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] = (
        r"C:\hadoop\bin" + os.pathsep
        + os.environ.get("PATH", "")
    )

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("THPT_2026_Analyze_Math")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 1. Doc du lieu da lam sach
df = spark.read.parquet(
    "data/processed/thpt_2026_clean.parquet"
)

# 2. Chi lay thi sinh co diem Toan hop le
math_df = df.filter(
    F.col("toan").isNotNull()
    & F.col("toan").between(0, 10)
)

print("\n=== THONG KE TOAN QUOC ===")

national = math_df.agg(
    F.count("*").alias("so_bai"),
    F.round(F.avg("toan"), 3).alias("diem_tb"),
    F.min("toan").alias("diem_min"),
    F.max("toan").alias("diem_max"),
    F.round(F.stddev_pop("toan"), 3).alias("do_lech_chuan"),
    F.expr("percentile_approx(toan, 0.5)").alias("trung_vi"),
    F.round(
        F.avg(F.when(F.col("toan") < 5, 1).otherwise(0)) * 100,
        2
    ).alias("ty_le_duoi_5"),
    F.round(
        F.avg(F.when(F.col("toan") >= 8, 1).otherwise(0)) * 100,
        2
    ).alias("ty_le_tu_8")
)

national.show(truncate=False)

# 3. Thong ke theo tinh/thanh
print("\n=== THONG KE THEO 34 TINH/THANH ===")

province_stats = math_df.groupBy(
    "province_code", "province_name"
).agg(
    F.count("*").alias("so_bai"),
    F.round(F.avg("toan"), 3).alias("diem_tb"),
    F.min("toan").alias("diem_min"),
    F.max("toan").alias("diem_max"),
    F.round(F.stddev_pop("toan"), 3).alias("do_lech_chuan"),
    F.expr("percentile_approx(toan, 0.5)").alias("trung_vi"),
    F.round(
        F.avg(F.when(F.col("toan") < 5, 1).otherwise(0)) * 100,
        2
    ).alias("ty_le_duoi_5"),
    F.round(
        F.avg(F.when(F.col("toan") >= 8, 1).otherwise(0)) * 100,
        2
    ).alias("ty_le_tu_8")
).orderBy(F.col("diem_tb").desc())

province_stats.show(34, truncate=False)

# 4. Pho diem theo khoang 1 diem
print("\n=== PHO DIEM TOAN TOAN QUOC ===")

distribution = math_df.withColumn(
    "khoang_diem",
    F.when(F.col("toan") == 10, F.lit(9))
     .otherwise(F.floor("toan"))
     .cast("int")
).groupBy("khoang_diem").agg(
    F.count("*").alias("so_bai")
).orderBy("khoang_diem")

distribution.show(10, truncate=False)

# 5. Pho diem theo tinh
province_distribution = math_df.withColumn(
    "khoang_diem",
    F.when(F.col("toan") == 10, F.lit(9))
     .otherwise(F.floor("toan"))
     .cast("int")
).groupBy(
    "province_code",
    "province_name",
    "khoang_diem"
).agg(
    F.count("*").alias("so_bai")
)

# 6. Luu ket qua
output_dir = "data/analysis"
os.makedirs(output_dir, exist_ok=True)

# Du lieu tong hop nho, xuat CSV bang Pandas
national.toPandas().to_csv(
    f"{output_dir}/math_national_stats.csv",
    index=False,
    encoding="utf-8-sig"
)

province_stats.toPandas().to_csv(
    f"{output_dir}/math_province_stats.csv",
    index=False,
    encoding="utf-8-sig"
)

distribution.toPandas().to_csv(
    f"{output_dir}/math_distribution.csv",
    index=False,
    encoding="utf-8-sig"
)

province_distribution.toPandas().to_csv(
    f"{output_dir}/math_province_distribution.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nDA HOAN THANH PHAN TICH PHO DIEM TOAN!")
print("Ket qua luu tai:", output_dir)

spark.stop()
