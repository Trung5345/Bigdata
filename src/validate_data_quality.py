
from pathlib import Path
import os
import sys

import pandas as pd

# ==================================================
# 1. CAU HINH WINDOWS
# ==================================================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

HADOOP = Path(r"C:\hadoop")

if (HADOOP / "bin").exists():
    os.environ["HADOOP_HOME"] = str(HADOOP)
    os.environ["PATH"] = (
        str(HADOOP / "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ==================================================
# 2. DUONG DAN
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed"
    / "thpt_2026_clean.parquet"
)

OUTPUT = (
    ROOT / "data" / "analysis"
    / "data_quality"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

SUBJECTS = {
    "toan": "Toán",
    "ngu_van": "Ngữ văn",
    "ngoai_ngu": "Ngoại ngữ",
    "vat_li": "Vật lý",
    "hoa_hoc": "Hóa học",
    "sinh_hoc": "Sinh học",
    "lich_su": "Lịch sử",
    "dia_li": "Địa lý",
    "gdkt_pl": "GD Kinh tế & Pháp luật",
    "tin_hoc": "Tin học",
    "cong_nghe_nn": "Công nghệ nông nghiệp",
    "cong_nghe_cn": "Công nghệ công nghiệp"
}


# ==================================================
# 3. HAM LUU CSV
# ==================================================

def save_csv(rows, filename):

    pd.DataFrame(rows).to_csv(
        OUTPUT / filename,
        index=False,
        encoding="utf-8-sig"
    )

    print("Da luu:", filename)


# ==================================================
# 4. KHOI TAO SPARK
# ==================================================

def create_spark():

    spark = (
        SparkSession.builder
        .appName("THPT_2026_Data_Quality")
        .master("local[4]")
        .config(
            "spark.driver.host", "127.0.0.1"
        )
        .config(
            "spark.driver.bindAddress",
            "127.0.0.1"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    return spark


# ==================================================
# 5. KIEM TRA TONG QUAN
# ==================================================

def check_overview(df, subjects):

    print("\n=== 1. TONG QUAN DU LIEU ===")

    total = df.count()

    province_count = (
        df.select("province_code")
        .distinct()
        .count()
    )

    print("Tong ban ghi:", f"{total:,}")
    print("So dia phuong:", province_count)
    print("So mon co cot diem:", len(subjects))

    rows = [
        {
            "chi_so": "tong_ban_ghi",
            "gia_tri": total
        },
        {
            "chi_so": "so_dia_phuong",
            "gia_tri": province_count
        },
        {
            "chi_so": "so_mon_co_cot_diem",
            "gia_tri": len(subjects)
        }
    ]

    save_csv(rows, "overview.csv")

    return total


# ==================================================
# 6. KIEM TRA TRUNG SO BAO DANH
# ==================================================

def check_duplicates(df):

    print("\n=== 2. KIEM TRA TRUNG SBD ===")

    if "sbd" not in df.columns:
        print(
            "Khong co cot sbd. Bo qua kiem tra."
        )
        return

    valid_sbd = df.filter(
        F.col("sbd").isNotNull()
        & (F.trim(F.col("sbd").cast("string")) != "")
    )

    duplicates = (
        valid_sbd
        .groupBy("province_code", "sbd")
        .count()
        .filter(F.col("count") > 1)
    )

    duplicate_groups = duplicates.count()

    duplicate_extra_rows = (
        duplicates
        .agg(
            F.sum(F.col("count") - 1)
            .alias("extra")
        )
        .first()["extra"]
        or 0
    )

    print(
        "Nhom SBD trung:",
        duplicate_groups
    )

    print(
        "Ban ghi trung du:",
        duplicate_extra_rows
    )

    rows = [
        row.asDict()
        for row in duplicates
        .orderBy(
            F.desc("count"),
            "province_code",
            "sbd"
        )
        .collect()
    ]

    save_csv(rows, "duplicate_sbd.csv")

    # Luu thong ke
    save_csv(
        [
            {
                "duplicate_groups": duplicate_groups,
                "duplicate_extra_rows": (
                    duplicate_extra_rows
                )
            }
        ],
        "duplicate_summary.csv"
    )


# ==================================================
# 7. KIEM TRA DIEM HOP LE
# ==================================================

def check_subjects(df, subjects, total):

    print("\n=== 3. KIEM TRA DIEM THEO MON ===")

    expressions = []

    for subject in subjects:

        col = F.col(subject).cast("double")

        expressions.extend([
            F.count(
                F.when(
                    col.isNotNull() & ~F.isnan(col),
                    1
                )
            ).alias(f"{subject}__valid_numeric"),

            F.count(
                F.when(
                    col.isNotNull()
                    & ~F.isnan(col)
                    & (
                        (col < 0)
                        | (col > 10)
                    ),
                    1
                )
            ).alias(f"{subject}__out_of_range"),

            F.count(
                F.when(
                    F.col(subject).isNotNull()
                    & (
                        col.isNull()
                        | F.isnan(col)
                    ),
                    1
                )
            ).alias(f"{subject}__non_numeric")
        ])

    stats = df.agg(*expressions).first().asDict()

    rows = []

    for subject in subjects:

        numeric = stats[
            f"{subject}__valid_numeric"
        ]

        out_range = stats[
            f"{subject}__out_of_range"
        ]

        non_numeric = stats[
            f"{subject}__non_numeric"
        ]

        usable = numeric - out_range

        rows.append({
            "mon_thi": subject,
            "ten_mon": SUBJECTS[subject],
            "tong_ban_ghi": total,
            "so_diem_hop_le": usable,
            "so_diem_ngoai_khoang": out_range,
            "so_gia_tri_khong_phai_so": non_numeric,
            "so_ban_ghi_khong_co_diem_hop_le": (
                total - usable
            ),
            "ty_le_co_diem_hop_le_pct": (
                round(usable / total * 100, 4)
                if total else 0
            )
        })

    print(
        pd.DataFrame(rows).to_string(
            index=False
        )
    )

    save_csv(rows, "subject_quality.csv")


# ==================================================
# 8. KIEM TRA SO BAN GHI THEO DIA PHUONG
# ==================================================

def check_provinces(df):

    print("\n=== 4. DU LIEU THEO DIA PHUONG ===")

    group_cols = ["province_code"]

    if "province_name" in df.columns:
        group_cols.append("province_name")

    province_stats = (
        df.groupBy(*group_cols)
        .agg(
            F.count("*").alias("so_ban_ghi")
        )
        .orderBy(F.desc("so_ban_ghi"))
    )

    rows = [
        r.asDict()
        for r in province_stats.collect()
    ]

    print(
        pd.DataFrame(rows).to_string(
            index=False
        )
    )

    save_csv(rows, "province_quality.csv")


# ==================================================
# 9. KIEM TRA GIA TRI DIEM PHO BIEN
# ==================================================

def check_score_patterns(df):

    print("\n=== 5. KIEM TRA TAN SUAT DIEM ===")

    selected_subjects = [
        s for s in [
            "toan", "ngu_van", "ngoai_ngu"
        ]
        if s in df.columns
    ]

    rows = []

    for subject in selected_subjects:

        score = F.col(subject).cast("double")

        top_scores = (
            df.filter(
                score.between(0, 10)
                & ~F.isnan(score)
            )
            .groupBy(
                score.alias("diem")
            )
            .count()
            .orderBy(F.desc("count"), "diem")
            .limit(15)
            .collect()
        )

        for item in top_scores:

            rows.append({
                "mon_thi": subject,
                "diem": item["diem"],
                "so_bai": item["count"]
            })

    save_csv(rows, "frequent_scores.csv")

    print(
        "Da tong hop 15 muc diem "
        "pho bien nhat moi mon duoc chon."
    )


# ==================================================
# 10. KIEM TRA THIEU DIEM THEO TINH
# ==================================================

def check_province_subjects(df, subjects):

    print("\n=== 6. THONG KE MON THEO TINH ===")

    expressions = [
        F.count("*").alias("tong_thi_sinh")
    ]

    for subject in subjects:

        score = F.col(subject).cast("double")

        expressions.append(
            F.count(
                F.when(
                    score.between(0, 10)
                    & ~F.isnan(score),
                    1
                )
            ).alias(subject)
        )

    grouped = (
        df.groupBy("province_code")
        .agg(*expressions)
        .orderBy("province_code")
    )

    rows = []

    for record in grouped.collect():

        record = record.asDict()

        for subject in subjects:

            total = record["tong_thi_sinh"]
            count = record[subject]

            rows.append({
                "province_code": (
                    record["province_code"]
                ),
                "mon_thi": subject,
                "ten_mon": SUBJECTS[subject],
                "tong_thi_sinh": total,
                "so_bai_co_diem": count,
                "ty_le_co_diem_pct": (
                    round(count / total * 100, 4)
                    if total else 0
                )
            })

    save_csv(
        rows,
        "province_subject_quality.csv"
    )

    print(
        "Da kiem tra:",
        len(rows),
        "cap tinh-mon."
    )


# ==================================================
# 11. MAIN
# ==================================================

def main():

    print("=" * 65)
    print("KIEM TRA CHAT LUONG DU LIEU THPT 2026")
    print("=" * 65)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Khong tim thay: {INPUT}"
        )

    spark = create_spark()

    try:

        df = spark.read.parquet(str(INPUT))

        if "province_code" not in df.columns:
            raise ValueError(
                "Du lieu thieu cot province_code."
            )

        subjects = [
            s for s in SUBJECTS
            if s in df.columns
        ]

        if not subjects:
            raise ValueError(
                "Khong tim thay cot diem mon thi."
            )

        df = df.cache()
        df.count()

        print("\nCac cot hien co:")
        print(df.columns)

        total = check_overview(
            df, subjects
        )

        check_duplicates(df)

        check_subjects(
            df, subjects, total
        )

        check_provinces(df)

        check_score_patterns(df)

        check_province_subjects(
            df, subjects
        )

        df.unpersist()

    finally:
        spark.stop()

    print("\n" + "=" * 65)
    print("HOAN THANH KIEM TRA DU LIEU")
    print("=" * 65)

    print("Thu muc ket qua:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
