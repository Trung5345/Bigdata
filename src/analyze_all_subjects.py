
import os
import sys

# ==========================================
# 1. CAU HINH PYSPARK TREN WINDOWS
# ==========================================

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
from pyspark import StorageLevel

spark = (
    SparkSession.builder
    .appName("THPT_2026_All_Subjects")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.python.worker.reuse", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# ==========================================
# 2. DUONG DAN DU LIEU
# ==========================================

INPUT_PATH = "data/processed/thpt_2026_clean.parquet"
OUTPUT_DIR = "data/analysis/all_subjects"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 3. DANH SACH 12 MON THI
# ==========================================

SUBJECTS = {
    "toan": "Toán",
    "ngu_van": "Ngữ văn",
    "ngoai_ngu": "Ngoại ngữ",
    "vat_li": "Vật lý",
    "hoa_hoc": "Hóa học",
    "sinh_hoc": "Sinh học",
    "lich_su": "Lịch sử",
    "dia_li": "Địa lý",
    "gdkt_pl": "Giáo dục kinh tế và pháp luật",
    "tin_hoc": "Tin học",
    "cong_nghe_nn": "Công nghệ nông nghiệp",
    "cong_nghe_cn": "Công nghệ công nghiệp"
}

# ==========================================
# 4. DOC DU LIEU DA LAM SACH
# ==========================================

df = spark.read.parquet(INPUT_PATH)

print("\n=== THONG TIN DU LIEU ===")
print("Tong so ban ghi:", df.count())

print(
    "Tong so tinh/thanh:",
    df.select("province_code").distinct().count()
)

# Kiem tra cot du lieu
required = [
    "province_code",
    "province_name"
] + list(SUBJECTS.keys())

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        "Thieu cac cot: " + ", ".join(missing)
    )

# ==========================================
# 5. CHUYEN DU LIEU SANG DANG LONG
# ==========================================

# Moi dong tuong ung 1 ket qua mon thi.
# Diem NULL la khong co ket qua hop le trong du lieu.

columns = [
    F.col("province_code").cast("string").alias(
        "province_code"
    ),
    F.col("province_name").alias("province_name")
]

score_structs = [
    F.struct(
        F.lit(code).alias("mon_thi"),
        F.col(code).cast("double").alias("diem")
    )
    for code in SUBJECTS.keys()
]

long_df = (
    df.select(
        *columns,
        F.explode(F.array(*score_structs)).alias("score")
    )
    .select(
        "province_code",
        "province_name",
        F.col("score.mon_thi").alias("mon_thi"),
        F.col("score.diem").alias("diem")
    )
    .filter(
        F.col("diem").isNotNull()
        & (~F.isnan("diem"))
        & F.col("diem").between(0, 10)
    )
)

# Tinh khoang diem, gom diem 10 vao nhom 9-10
long_df = long_df.withColumn(
    "khoang_diem",
    F.when(
        F.col("diem") == 10,
        F.lit(9)
    ).otherwise(
        F.floor("diem")
    ).cast("int")
)

# Luu cache de tai su dung nhieu phep thong ke
long_df = long_df.persist(
    StorageLevel.MEMORY_AND_DISK
)

print("\nSo ket qua mon thi hop le:", long_df.count())

# ==========================================
# 6. HAM THONG KE
# ==========================================

def statistic_expressions():
    return [
        F.count("*").alias("so_bai"),
        F.round(F.avg("diem"), 3).alias("diem_tb"),
        F.min("diem").alias("diem_min"),
        F.max("diem").alias("diem_max"),
        F.round(
            F.stddev_pop("diem"), 3
        ).alias("do_lech_chuan"),
        F.expr(
            "percentile_approx(diem, 0.5)"
        ).alias("trung_vi"),
        F.round(
            F.avg(
                F.when(
                    F.col("diem") < 5, 1
                ).otherwise(0)
            ) * 100, 2
        ).alias("ty_le_duoi_5"),
        F.round(
            F.avg(
                F.when(
                    F.col("diem") >= 8, 1
                ).otherwise(0)
            ) * 100, 2
        ).alias("ty_le_tu_8")
    ]

# ==========================================
# 7. THONG KE TOAN QUOC THEO MON
# ==========================================

print("\n=== THONG KE TOAN QUOC 12 MON ===")

national_stats = (
    long_df.groupBy("mon_thi")
    .agg(*statistic_expressions())
    .orderBy("mon_thi")
)

national_stats.show(20, truncate=False)

# ==========================================
# 8. THONG KE THEO MON VA TINH
# ==========================================

print("\n=== THONG KE THEO TINH ===")

province_stats = (
    long_df.groupBy(
        "mon_thi",
        "province_code",
        "province_name"
    )
    .agg(*statistic_expressions())
    .orderBy(
        "mon_thi",
        F.col("diem_tb").desc()
    )
)

print("So nhom mon-tinh:", province_stats.count())

# ==========================================
# 9. PHO DIEM TOAN QUOC THEO MON
# ==========================================

national_distribution = (
    long_df.groupBy(
        "mon_thi",
        "khoang_diem"
    )
    .agg(
        F.count("*").alias("so_bai")
    )
    .orderBy("mon_thi", "khoang_diem")
)

# ==========================================
# 10. PHO DIEM THEO MON VA TINH
# ==========================================

province_distribution = (
    long_df.groupBy(
        "mon_thi",
        "province_code",
        "province_name",
        "khoang_diem"
    )
    .agg(
        F.count("*").alias("so_bai")
    )
    .orderBy(
        "mon_thi",
        "province_code",
        "khoang_diem"
    )
)

# ==========================================
# 11. HAM XUAT CSV
# ==========================================

def save_csv(dataframe, filename):
    path = os.path.join(OUTPUT_DIR, filename)

    # Chi thu thap bang da tong hop ve driver
    pdf = dataframe.toPandas()

    # Bao toan ma tinh co so 0 o dau
    if "province_code" in pdf.columns:
        pdf["province_code"] = (
            pdf["province_code"]
            .astype(str)
            .str.zfill(2)
        )

    pdf.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Da luu:", path)

# ==========================================
# 12. LUU KET QUA
# ==========================================

print("\n=== DANG XUAT KET QUA ===")

save_csv(
    national_stats,
    "all_national_stats.csv"
)

save_csv(
    province_stats,
    "all_province_stats.csv"
)

save_csv(
    national_distribution,
    "all_national_distribution.csv"
)

save_csv(
    province_distribution,
    "all_province_distribution.csv"
)

# ==========================================
# 13. HOAN THANH
# ==========================================

long_df.unpersist()

print("\n====================================")
print("HOAN THANH PHAN TICH 12 MON THI!")
print("Thu muc ket qua:", OUTPUT_DIR)
print("====================================")

spark.stop()
