
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CAU HINH PYSPARK TREN WINDOWS
# ==========================================

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
    .appName("THPT_2026_Literature_Spikes")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# ==========================================
# 2. CAU HINH DU LIEU
# ==========================================

INPUT_PATH = "data/processed/thpt_2026_clean.parquet"
OUTPUT_DIR = "data/analysis/score_spikes"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROVINCES = {
    "86": "Vĩnh Long",
    "80": "Tây Ninh",
    "14": "Sơn La"
}

# ==========================================
# 3. DOC DIEM NGU VAN
# ==========================================

df = spark.read.parquet(INPUT_PATH)

df = (
    df.select(
        F.lpad(
            F.col("province_code").cast("string"),
            2, "0"
        ).alias("province_code"),
        F.col("province_name"),
        F.col("ngu_van").cast("double").alias("diem")
    )
    .filter(
        F.col("diem").isNotNull()
        & F.col("diem").between(0, 10)
    )
)

print("Tong so bai Ngu van:", df.count())

# ==========================================
# 4. PHAN TICH THEO MUC DIEM 0.25
# ==========================================

# Giu lai cac diem nam tren luoi 0.25.
# Khong lam tron diem 0.1 thanh diem 0.25,
# vi nhu vay co the tao spike gia.
df = df.withColumn(
    "diem_x4",
    F.round(F.col("diem") * 4).cast("int")
)

df = df.withColumn(
    "on_grid_025",
    F.abs(
        F.col("diem") * 4 - F.col("diem_x4")
    ) < 0.000001
)

non_grid = df.filter(~F.col("on_grid_025")).count()
print("So bai khong nam tren luoi 0.25:", non_grid)

grid_df = df.filter(F.col("on_grid_025"))

# Tan suat moi muc diem theo tinh
province_freq = (
    grid_df.groupBy(
        "province_code",
        "province_name",
        "diem_x4"
    )
    .agg(F.count("*").alias("so_bai"))
    .withColumn(
        "diem",
        F.col("diem_x4") / 4.0
    )
)

# Tan suat toan quoc
national_freq = (
    grid_df.groupBy("diem_x4")
    .agg(F.count("*").alias("so_bai"))
    .withColumn(
        "diem",
        F.col("diem_x4") / 4.0
    )
)

province_freq.toPandas().to_csv(
    f"{OUTPUT_DIR}/literature_province_025.csv",
    index=False,
    encoding="utf-8-sig"
)

national_freq.toPandas().to_csv(
    f"{OUTPUT_DIR}/literature_national_025.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 5. KIEM TRA DIEM 4.75, 5.00, 5.25
# ==========================================

rows = []

# Chi so dia phuong can kiem tra
for code, name in PROVINCES.items():

    subset = (
        province_freq
        .filter(F.col("province_code") == code)
        .filter(F.col("diem_x4").isin(19, 20, 21))
        .select("diem_x4", "so_bai")
        .collect()
    )

    counts = {
        int(row["diem_x4"]): int(row["so_bai"])
        for row in subset
    }

    n475 = counts.get(19, 0)
    n500 = counts.get(20, 0)
    n525 = counts.get(21, 0)

    expected = (n475 + n525) / 2

    spike_ratio = (
        n500 / expected
        if expected > 0 else np.nan
    )

    rows.append({
        "province_code": code,
        "province_name": name,
        "diem_475": n475,
        "diem_500": n500,
        "diem_525": n525,
        "muc_noi_suy": expected,
        "spike_ratio": spike_ratio,
        "so_bai_doi_ra": n500 - expected
    })

result = pd.DataFrame(rows)

print("\n=== KIEM TRA SPIKE TAI DIEM 5.0 ===")
print(result.to_string(index=False))

result.to_csv(
    f"{OUTPUT_DIR}/literature_spike_5.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 6. VE BIEU DO PHO DIEM CHI TIET
# ==========================================

freq_pdf = province_freq.filter(
    F.col("province_code").isin(
        list(PROVINCES.keys())
    )
).toPandas()

national_pdf = national_freq.toPandas()

fig, ax = plt.subplots(figsize=(13, 6))

x = np.arange(0, 10.25, 0.25)

for code, name in PROVINCES.items():
    temp = freq_pdf[
        freq_pdf["province_code"] == code
    ]

    counts = (
        temp.set_index("diem")
        ["so_bai"]
        .reindex(x, fill_value=0)
    )

    total = counts.sum()

    if total > 0:
        ax.plot(
            x,
            counts / total * 100,
            marker=".",
            linewidth=1.5,
            label=name
        )

national_counts = (
    national_pdf.set_index("diem")["so_bai"]
    .reindex(x, fill_value=0)
)

ax.plot(
    x,
    national_counts / national_counts.sum() * 100,
    linestyle="--",
    linewidth=2,
    label="Toàn quốc"
)

ax.axvline(
    5.0,
    linestyle=":",
    linewidth=1.5,
    label="Mức điểm 5.0"
)

ax.set_xlim(2, 9)
ax.set_xlabel("Điểm Ngữ văn")
ax.set_ylabel("Tỷ lệ bài thi trên lưới 0.25 (%)")
ax.set_title(
    "Phân bố điểm Ngữ văn THPT 2026"
    " theo mức 0.25 điểm"
)

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

chart_path = (
    f"{OUTPUT_DIR}/literature_spike_comparison.png"
)

plt.savefig(chart_path, dpi=180)
plt.show()

print("\nDa luu bieu do:", chart_path)
print("Da luu bang thong ke tai:", OUTPUT_DIR)

spark.stop()
