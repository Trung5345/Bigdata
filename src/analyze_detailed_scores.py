
from pathlib import Path
import os
import sys
import pandas as pd

# =====================================================
# 1. CAU HINH WINDOWS
# =====================================================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

HADOOP = Path(r"C:\hadoop")

if (HADOOP / "bin").is_dir():
    os.environ["HADOOP_HOME"] = str(HADOOP)
    os.environ["PATH"] = (
        str(HADOOP / "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# =====================================================
# 2. DUONG DAN VA THAM SO
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed"
    / "thpt_2026_clean.parquet"
)

OUTPUT = (
    ROOT / "data" / "analysis"
    / "detailed_scores"
)

SUBJECTS = {
    "toan": "Toán",
    "ngu_van": "Ngữ văn",
    "ngoai_ngu": "Ngoại ngữ",
    "vat_li": "Vật lý",
    "hoa_hoc": "Hóa học",
    "sinh_hoc": "Sinh học",
    "lich_su": "Lịch sử",
    "dia_li": "Địa lý",
    "gdkt_pl": "Giáo dục KT&PL",
    "tin_hoc": "Tin học",
    "cong_nghe_nn": "Công nghệ nông nghiệp",
    "cong_nghe_cn": "Công nghệ công nghiệp"
}

STEP = 0.25
EPS = 1e-7


# =====================================================
# 3. KHOI TAO SPARK
# =====================================================

def create_spark():

    spark = (
        SparkSession.builder
        .appName("THPT_Detailed_Scores_V2")
        .master("local[4]")
        .config(
            "spark.driver.host",
            "127.0.0.1"
        )
        .config(
            "spark.driver.bindAddress",
            "127.0.0.1"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    return spark


# =====================================================
# 4. TINH SPIKE RATIO
# =====================================================

def calculate_spikes(province_exact):

    rows = []

    groups = province_exact.groupby(
        [
            "mon_thi",
            "province_code",
            "province_name"
        ],
        sort=False
    )

    for (subject, code, name), group in groups:

        frequencies = {
            int(round(float(row.diem) * 4)):
                int(row.so_bai)
            for row in group.itertuples(index=False)
        }

        # Chi xet cac diem ben trong (0, 10)
        for i in range(1, 40):

            left = frequencies.get(i - 1, 0)
            current = frequencies.get(i, 0)
            right = frequencies.get(i + 1, 0)

            neighbor_mean = (left + right) / 2

            if neighbor_mean <= 0:
                continue

            rows.append({
                "mon_thi": subject,
                "province_code": code,
                "province_name": name,
                "diem": i * STEP,
                "so_bai_tai_diem": current,
                "so_bai_ben_trai": left,
                "so_bai_ben_phai": right,
                "trung_binh_lan_can": neighbor_mean,
                "spike_ratio": (
                    current / neighbor_mean
                )
            })

    return pd.DataFrame(
        rows,
        columns=[
            "mon_thi",
            "province_code",
            "province_name",
            "diem",
            "so_bai_tai_diem",
            "so_bai_ben_trai",
            "so_bai_ben_phai",
            "trung_binh_lan_can",
            "spike_ratio"
        ]
    )


# =====================================================
# 5. CHUONG TRINH CHINH
# =====================================================

def main():

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    spark = create_spark()

    national_bins = []
    province_bins = []

    national_exact = []
    province_exact = []

    audits = []

    try:

        df = spark.read.parquet(
            str(INPUT)
        )

        required = {
            "province_code",
            "province_name"
        }

        if not required.issubset(set(df.columns)):
            raise ValueError(
                "Du lieu thieu thong tin dia phuong."
            )

        for subject, name in SUBJECTS.items():

            if subject not in df.columns:
                raise ValueError(
                    f"Thieu cot diem mon: {subject}"
                )

            print(
                f"\nDang xu ly: {name}",
                flush=True
            )

            score = F.col(subject).cast("double")

            # Lay tat ca diem hop le 0-10
            valid = (
                df.select(
                    "province_code",
                    "province_name",
                    score.alias("diem")
                )
                .filter(
                    F.col("diem").isNotNull()
                    & ~F.isnan("diem")
                    & F.col("diem").between(0, 10)
                )
            )

            # =====================================
            # A. KHOANG DIEM 0.25
            #
            # [0,0.25), [0.25,0.5) ...
            # Khoang cuoi [9.75,10]
            # =====================================

            binned = valid.withColumn(
                "bin_index",
                F.least(
                    F.floor(
                        F.col("diem") * 4
                    ).cast("int"),
                    F.lit(39)
                )
            )

            nat_bin = (
                binned.groupBy("bin_index")
                .count()
                .withColumnRenamed(
                    "count", "so_bai"
                )
                .withColumn(
                    "mon_thi",
                    F.lit(subject)
                )
            )

            prov_bin = (
                binned.groupBy(
                    "province_code",
                    "province_name",
                    "bin_index"
                )
                .count()
                .withColumnRenamed(
                    "count", "so_bai"
                )
                .withColumn(
                    "mon_thi",
                    F.lit(subject)
                )
            )

            # =====================================
            # B. DIEM CHINH XAC TREN LUOI 0.25
            #
            # Khong lam tron diem ngoai luoi
            # =====================================

            positions = valid.withColumn(
                "grid_index",
                F.round(
                    F.col("diem") * 4
                ).cast("int")
            )

            exact = (
                positions.filter(
                    F.abs(
                        F.col("diem")
                        - F.col("grid_index") / 4.0
                    ) <= EPS
                )
                .withColumn(
                    "diem",
                    F.col("grid_index") / 4.0
                )
            )

            nat_exact = (
                exact.groupBy("diem")
                .count()
                .withColumnRenamed(
                    "count", "so_bai"
                )
                .withColumn(
                    "mon_thi",
                    F.lit(subject)
                )
            )

            prov_exact = (
                exact.groupBy(
                    "province_code",
                    "province_name",
                    "diem"
                )
                .count()
                .withColumnRenamed(
                    "count", "so_bai"
                )
                .withColumn(
                    "mon_thi",
                    F.lit(subject)
                )
            )

            # Chuyen bang tong hop nho ve Pandas
            nb = nat_bin.toPandas()
            pb = prov_bin.toPandas()

            ne = nat_exact.toPandas()
            pe = prov_exact.toPandas()

            total_valid = int(
                nb["so_bai"].sum()
            )

            total_grid = int(
                ne["so_bai"].sum()
            )

            # Kiem tra so bai giua cac bang
            if int(pb["so_bai"].sum()) != total_valid:
                raise AssertionError(
                    f"Lech tong pho diem tinh: {subject}"
                )

            if int(pe["so_bai"].sum()) != total_grid:
                raise AssertionError(
                    f"Lech tong diem luoi: {subject}"
                )

            national_bins.append(nb)
            province_bins.append(pb)

            national_exact.append(ne)
            province_exact.append(pe)

            audits.append({
                "mon_thi": subject,
                "ten_mon": name,
                "tong_diem_hop_le": total_valid,
                "so_diem_tren_luoi": total_grid,
                "so_diem_ngoai_luoi": (
                    total_valid - total_grid
                ),
                "ty_le_tren_luoi_pct": (
                    100 * total_grid / total_valid
                    if total_valid else 0
                ),
                "so_bai_trong_cac_khoang":
                    total_valid
            })

            print(
                f"  Pho diem theo khoang: "
                f"{total_valid:,} bai"
            )

            print(
                f"  Diem tren luoi 0.25: "
                f"{total_grid:,} bai"
            )

    finally:
        spark.stop()

    # ===========================================
    # 6. GHEP KET QUA 12 MON
    # ===========================================

    nat_bins_df = pd.concat(
        national_bins,
        ignore_index=True
    )

    prov_bins_df = pd.concat(
        province_bins,
        ignore_index=True
    )

    nat_exact_df = pd.concat(
        national_exact,
        ignore_index=True
    )

    prov_exact_df = pd.concat(
        province_exact,
        ignore_index=True
    )

    # Them gioi han cua moi khoang
    for frame in [nat_bins_df, prov_bins_df]:

        frame["bin_index"] = (
            frame["bin_index"].astype(int)
        )

        frame["diem_tu"] = (
            frame["bin_index"] * STEP
        )

        frame["diem_den"] = (
            (frame["bin_index"] + 1) * STEP
        )

        frame["nhan_khoang"] = frame.apply(
            lambda row: (
                f"[{row['diem_tu']:.2f}, "
                f"{row['diem_den']:.2f}"
                f"{']' if row['bin_index'] == 39 else ')'}"
            ),
            axis=1
        )

    # Sap xep
    nat_bins_df = nat_bins_df.sort_values(
        ["mon_thi", "bin_index"]
    )

    prov_bins_df = prov_bins_df.sort_values(
        [
            "mon_thi",
            "province_code",
            "bin_index"
        ]
    )

    nat_exact_df = nat_exact_df.sort_values(
        ["mon_thi", "diem"]
    )

    prov_exact_df = prov_exact_df.sort_values(
        [
            "mon_thi",
            "province_code",
            "diem"
        ]
    )

    # ===========================================
    # 7. TINH SPIKE RATIO
    # ===========================================

    spike_df = calculate_spikes(
        prov_exact_df
    )

    audit_df = pd.DataFrame(audits)

    # ===========================================
    # 8. XUAT CSV
    # ===========================================

    outputs = {
        "all_binned_national.csv":
            nat_bins_df,

        "all_binned_province.csv":
            prov_bins_df,

        "all_detailed_national.csv":
            nat_exact_df,

        "all_detailed_province.csv":
            prov_exact_df,

        "all_detailed_spikes.csv":
            spike_df,

        "all_detailed_audit.csv":
            audit_df
    }

    for filename, data in outputs.items():

        data.to_csv(
            OUTPUT / filename,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"Da luu {filename}: "
            f"{len(data):,} dong"
        )

    # ===========================================
    # 9. KIEM TRA CUOI
    # ===========================================

    print("\n" + "=" * 65)
    print("TONG HOP DOI CHIEU")
    print("=" * 65)

    print(
        audit_df[
            [
                "ten_mon",
                "tong_diem_hop_le",
                "so_bai_trong_cac_khoang",
                "ty_le_tren_luoi_pct"
            ]
        ].to_string(index=False)
    )

    print("\nHOAN THANH!")


if __name__ == "__main__":
    main()
