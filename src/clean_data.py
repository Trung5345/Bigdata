
import os
import sys
import html
import unicodedata

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import MapType, StringType

# ==================================================
# 1. CAU HINH PYSPARK TREN WINDOWS
# ==================================================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

# Neu may da co Hadoop tai C:\hadoop
if os.path.isdir(r"C:\hadoop\bin"):
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] = (
        r"C:\hadoop\bin" + os.pathsep
        + os.environ.get("PATH", "")
    )

spark = (
    SparkSession.builder
    .appName("THPT_2026_Clean_Data")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.python.worker.reuse", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# ==================================================
# 2. DUONG DAN DU LIEU
# ==================================================

INPUT_PATH = "data/processed/thpt_2026.parquet"
OUTPUT_PATH = "data/processed/thpt_2026_clean.parquet"

# Doc du lieu goc da gop tu 34 file CSV
df = spark.read.parquet(INPUT_PATH)

print("=== THONG TIN DU LIEU BAN DAU ===")
print("Tong so ban ghi:", df.count())
print("So tinh/thanh:",
      df.select("province_code").distinct().count())

# ==================================================
# 3. DANH SACH MON THI
# ==================================================

SCORE_COLUMNS = [
    "toan",
    "ngu_van",
    "ngoai_ngu",
    "vat_li",
    "hoa_hoc",
    "sinh_hoc",
    "lich_su",
    "dia_li",
    "gdkt_pl",
    "tin_hoc",
    "cong_nghe_nn",
    "cong_nghe_cn"
]

# Tao cac cot chua ton tai
for subject in SCORE_COLUMNS:
    if subject not in df.columns:
        df = df.withColumn(
            subject,
            F.lit(None).cast("string")
        )

# ==================================================
# 4. CHUAN HOA TEN MON HOC
# ==================================================

def normalize_subject(name):
    if not name:
        return ""

    name = unicodedata.normalize("NFD", name)

    name = "".join(
        char for char in name
        if unicodedata.category(char) != "Mn"
    )

    name = name.lower().strip()
    name = name.replace("đ", "d")
    name = " ".join(name.split())

    return name


# Anh xa ten mon hoc sang cot du lieu
SUBJECT_MAP = {
    "toan": "toan",

    "van": "ngu_van",
    "ngu van": "ngu_van",

    "ngoai ngu": "ngoai_ngu",
    "anh": "ngoai_ngu",
    "tieng anh": "ngoai_ngu",

    "li": "vat_li",
    "ly": "vat_li",
    "vat li": "vat_li",
    "vat ly": "vat_li",

    "hoa": "hoa_hoc",
    "hoa hoc": "hoa_hoc",

    "sinh": "sinh_hoc",
    "sinh hoc": "sinh_hoc",

    "su": "lich_su",
    "lich su": "lich_su",

    "dia": "dia_li",
    "dia li": "dia_li",
    "dia ly": "dia_li",

    "giao duc kinh te phap luat": "gdkt_pl",
    "giao duc kinh te va phap luat": "gdkt_pl",
    "gdktpl": "gdkt_pl",

    "tin": "tin_hoc",
    "tin hoc": "tin_hoc",

    "cong nghe nong nghiep": "cong_nghe_nn",
    "cong nghe cong nghiep": "cong_nghe_cn"
}

# ==================================================
# 5. GIAI MA VA TACH DIEM TRONG EXTRA
# ==================================================

def parse_extra(value):
    result = {}

    if value is None or not value.strip():
        return result

    decoded = html.unescape(value)

    for item in decoded.split(";"):
        item = item.strip()

        if "=" not in item:
            continue

        subject_name, score_text = item.split("=", 1)

        subject_key = normalize_subject(subject_name)
        column_name = SUBJECT_MAP.get(subject_key)

        if column_name is None:
            continue

        try:
            score = float(score_text.strip())

            # Diem thi hop le: 0 den 10
            if 0 <= score <= 10:
                result[column_name] = str(score)

        except (ValueError, TypeError):
            continue

    return result


parse_extra_udf = F.udf(
    parse_extra,
    MapType(StringType(), StringType())
)

df = df.withColumn(
    "parsed_extra",
    parse_extra_udf(F.col("extra"))
)

# ==================================================
# 6. CHUAN HOA DIEM THI
# ==================================================

print("\n=== CHUAN HOA DIEM THI ===")

# Dem so diem duoc bo sung va so diem mau thuan
filled_expressions = []
conflict_expressions = []

for subject in SCORE_COLUMNS:

    # Gia tri ban dau trong CSV
    original_text = F.trim(F.col(subject))

    original_score = (
        F.when(
            original_text == "",
            F.lit(None)
        )
        .otherwise(original_text)
        .cast("double")
    )

    # Gia tri lay tu extra
    extra_score = (
        F.col("parsed_extra")
        .getItem(subject)
        .cast("double")
    )

    # Danh dau truong hop bo sung diem
    filled_expressions.append(
        F.sum(
            F.when(
                original_score.isNull()
                & extra_score.isNotNull(),
                1
            ).otherwise(0)
        ).alias("filled_" + subject)
    )

    # Danh dau diem mau thuan
    conflict_expressions.append(
        F.sum(
            F.when(
                original_score.isNotNull()
                & extra_score.isNotNull()
                & (F.abs(original_score - extra_score) > 0.000001),
                1
            ).otherwise(0)
        ).alias("conflict_" + subject)
    )

    # Uu tien diem goc, chi bo sung neu thieu
    df = df.withColumn(
        subject,
        F.coalesce(
            original_score,
            extra_score
        )
    )

# ==================================================
# 7. KIEM TRA CHAT LUONG DU LIEU
# ==================================================

print("\n=== KIEM TRA DU LIEU ===")

# Dem tong so ban ghi
total_records = df.count()

# Dem so tinh/thanh
total_provinces = (
    df.select("province_code")
    .distinct()
    .count()
)

print("Tong so thi sinh:", total_records)
print("Tong so tinh/thanh:", total_provinces)

# Thong ke so diem hop le va ngoai khoang
stats_expr = []

for subject in SCORE_COLUMNS:
    stats_expr.extend([
        F.count(subject).alias(subject + "_count"),

        F.sum(
            F.when(
                F.col(subject).isNotNull()
                & (
                    (F.col(subject) < 0)
                    | (F.col(subject) > 10)
                    | F.isnan(F.col(subject))
                ),
                1
            ).otherwise(0)
        ).alias(subject + "_invalid")
    ])

stats = df.agg(*stats_expr).first()

print("\nSO BAI THI THEO MON:")

for subject in SCORE_COLUMNS:
    print(
        f"{subject}: "
        f"{stats[subject + '_count']} bai, "
        f"{stats[subject + '_invalid']} diem khong hop le"
    )

# ==================================================
# 8. KIEM TRA DIEM DUOC BO SUNG VA MAU THUAN
# ==================================================

# Tinh lai thong ke bang cach doi chieu du lieu goc
# voi gia tri tu extra

original_df = spark.read.parquet(INPUT_PATH)

for subject in SCORE_COLUMNS:
    if subject not in original_df.columns:
        original_df = original_df.withColumn(
            subject,
            F.lit(None).cast("string")
        )

original_df = original_df.withColumn(
    "parsed_extra",
    parse_extra_udf(F.col("extra"))
)

check_expr = []

for subject in SCORE_COLUMNS:
    original_value = F.when(
        F.trim(F.col(subject)) == "",
        F.lit(None)
    ).otherwise(
        F.trim(F.col(subject))
    ).cast("double")

    extra_value = (
        F.col("parsed_extra")
        .getItem(subject)
        .cast("double")
    )

    check_expr.append(
        F.sum(
            F.when(
                original_value.isNull()
                & extra_value.isNotNull(),
                1
            ).otherwise(0)
        ).alias("filled_" + subject)
    )

    check_expr.append(
        F.sum(
            F.when(
                original_value.isNotNull()
                & extra_value.isNotNull()
                & (F.abs(original_value - extra_value) > 0.000001),
                1
            ).otherwise(0)
        ).alias("conflict_" + subject)
    )

check_result = original_df.agg(*check_expr).first()

print("\nSO DIEM DUOC BO SUNG TU EXTRA:")

for subject in SCORE_COLUMNS:
    print(
        f"{subject}: "
        f"{check_result['filled_' + subject]} diem bo sung, "
        f"{check_result['conflict_' + subject]} diem mau thuan"
    )

# ==================================================
# 9. HIEN THI DU LIEU SAU LAM SACH
# ==================================================

print("\n=== 15 BAN GHI SAU KHI LAM SACH ===")

df.select(
    "sbd",
    "province_name",
    "toan",
    "ngu_van",
    "vat_li",
    "hoa_hoc",
    "lich_su",
    "dia_li",
    "gdkt_pl",
    "cong_nghe_nn",
    "cong_nghe_cn"
).show(15, truncate=False)

# ==================================================
# 10. LUU DU LIEU PARQUET
# ==================================================

# Giu lai cot extra de doi chieu sau nay
df = df.drop("parsed_extra")

df.write.mode("overwrite").parquet(OUTPUT_PATH)

print("\n===================================")
print("DA HOAN THANH CHUAN HOA DU LIEU!")
print("Duong dan:", OUTPUT_PATH)
print("===================================")

spark.stop()
