
from pathlib import Path
from itertools import combinations
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

# =====================================================
# 1. CAU HINH WINDOWS
# =====================================================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

HADOOP_HOME = Path(r"C:\hadoop")

if (HADOOP_HOME / "bin").exists():
    os.environ["HADOOP_HOME"] = str(HADOOP_HOME)
    os.environ["PATH"] = (
        str(HADOOP_HOME / "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import ClusteringEvaluator


# =====================================================
# 2. DUONG DAN
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DISTRIBUTION = (
    ROOT / "data" / "analysis" / "all_subjects"
    / "all_province_distribution.csv"
)

INPUT_KMEANS = (
    ROOT / "data" / "analysis" / "all_subjects"
    / "all_kmeans_results.csv"
)

OUTPUT = (
    ROOT / "data" / "analysis" / "kmeans_stability"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]

MIN_SAMPLES = 500
MIN_PROVINCES_PER_CLUSTER = 3

FEATURE_COLS = [f"bin_{i}" for i in range(10)]


# =====================================================
# 3. TEN MON THI
# =====================================================

SUBJECT_NAMES = {
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


# =====================================================
# 4. DOC VA CHUAN BI DU LIEU
# =====================================================

def prepare_features():

    print("\nDANG DOC DU LIEU PHO DIEM...")

    distribution = pd.read_csv(
        INPUT_DISTRIBUTION,
        dtype={"province_code": str}
    )

    original_kmeans = pd.read_csv(
        INPUT_KMEANS,
        dtype={"province_code": str}
    )

    distribution["province_code"] = (
        distribution["province_code"].str.zfill(2)
    )

    original_kmeans["province_code"] = (
        original_kmeans["province_code"].str.zfill(2)
    )

    # Chi lay cac mon da co ket qua K-means
    subjects = sorted(
        original_kmeans["mon_thi"].unique()
    )

    # Pivot: moi tinh la 1 dong,
    # 10 khoang diem la 10 cot
    pivot = distribution.pivot_table(
        index=[
            "mon_thi",
            "province_code",
            "province_name"
        ],
        columns="khoang_diem",
        values="so_bai",
        aggfunc="sum",
        fill_value=0
    )

    pivot = pivot.reindex(
        columns=range(10),
        fill_value=0
    )

    pivot.columns = FEATURE_COLS
    pivot = pivot.reset_index()

    pivot["so_bai"] = (
        pivot[FEATURE_COLS].sum(axis=1)
    )

    pivot = pivot[
        (pivot["mon_thi"].isin(subjects))
        & (pivot["so_bai"] >= MIN_SAMPLES)
    ].copy()

    # Chuyen so bai moi khoang thanh ty le
    for col in FEATURE_COLS:
        pivot[col] = (
            pivot[col] / pivot["so_bai"]
        )

    return pivot, original_kmeans


# =====================================================
# 5. TAO SPARK SESSION
# =====================================================

def create_spark():

    spark = (
        SparkSession.builder
        .appName("THPT_KMeans_Stability")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    return spark


# =====================================================
# 6. PHAN CUM MOT MON
# =====================================================

def evaluate_subject(spark, subject_df, subject, k):

    print("\n" + "=" * 60)
    print("MON:", SUBJECT_NAMES.get(subject, subject))
    print("K =", k)
    print("SO DIA PHUONG =", len(subject_df))
    print("=" * 60)

    # Tao Spark DataFrame
    sdf = spark.createDataFrame(subject_df)

    # Vector hoa 10 dac trung
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="raw_features"
    )

    sdf = assembler.transform(sdf)

    # Chuan hoa dac trung
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True
    )

    scaler_model = scaler.fit(sdf)
    sdf = scaler_model.transform(sdf)

    # Cache du lieu vi train nhieu lan
    sdf = sdf.cache()
    sdf.count()

    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="prediction",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )

    runs = []
    labels_by_seed = {}

    try:
        for seed in SEEDS:

            model = KMeans(
                k=k,
                seed=seed,
                featuresCol="features",
                predictionCol="prediction",
                maxIter=100
            ).fit(sdf)

            predictions = model.transform(sdf).cache()

            try:
                silhouette = float(
                    evaluator.evaluate(predictions)
                )

                rows = (
                    predictions
                    .select("province_code", "prediction")
                    .orderBy("province_code")
                    .collect()
                )

                labels = [
                    int(row["prediction"])
                    for row in rows
                ]

                cluster_sizes = (
                    pd.Series(labels)
                    .value_counts()
                    .sort_index()
                    .to_dict()
                )

                min_size = min(cluster_sizes.values())

                valid = (
                    len(cluster_sizes) == k
                    and min_size >= MIN_PROVINCES_PER_CLUSTER
                )

                labels_by_seed[seed] = labels

                runs.append({
                    "mon_thi": subject,
                    "ten_mon": SUBJECT_NAMES.get(
                        subject, subject
                    ),
                    "seed": seed,
                    "k": k,
                    "so_dia_phuong": len(labels),
                    "silhouette": silhouette,
                    "so_cum_thuc_te": len(cluster_sizes),
                    "kich_thuoc_cum_nho_nhat": min_size,
                    "cum_hop_le": valid,
                    "cluster_sizes": str(cluster_sizes)
                })

                print(
                    f"Seed {seed}: "
                    f"Silhouette={silhouette:.4f} | "
                    f"Sizes={cluster_sizes} | "
                    f"Hop le={valid}"
                )

            finally:
                predictions.unpersist()

    finally:
        sdf.unpersist()

    # ARI giua moi cap seed
    ari_values = []

    for seed_a, seed_b in combinations(SEEDS, 2):

        ari = adjusted_rand_score(
            labels_by_seed[seed_a],
            labels_by_seed[seed_b]
        )

        ari_values.append(float(ari))

    run_df = pd.DataFrame(runs)

    return run_df, ari_values


# =====================================================
# 7. HAM MAIN
# =====================================================

def main():

    if not INPUT_DISTRIBUTION.exists():
        raise FileNotFoundError(
            f"Thieu file: {INPUT_DISTRIBUTION}"
        )

    if not INPUT_KMEANS.exists():
        raise FileNotFoundError(
            f"Thieu file: {INPUT_KMEANS}"
        )

    features, original = prepare_features()

    print("Tong so mon:", features["mon_thi"].nunique())
    print("Tong so dong:", len(features))

    spark = create_spark()

    all_runs = []
    summaries = []

    try:

        for subject in sorted(
            original["mon_thi"].unique()
        ):

            data = features[
                features["mon_thi"] == subject
            ].copy()

            if data.empty:
                continue

            original_subject = original[
                original["mon_thi"] == subject
            ]

            # Lay K da chon trong ket qua cu
            k = int(
                original_subject["best_k"].iloc[0]
            )

            if len(data) < k * MIN_PROVINCES_PER_CLUSTER:

                print(
                    "BO QUA:",
                    subject,
                    "- Khong du dia phuong"
                )

                continue

            # Chi giu cac cot can thiet
            cols = [
                "province_code",
                "province_name",
                *FEATURE_COLS
            ]

            run_df, ari_values = evaluate_subject(
                spark,
                data[cols],
                subject,
                k
            )

            all_runs.append(run_df)

            summaries.append({
                "mon_thi": subject,
                "ten_mon": SUBJECT_NAMES.get(
                    subject, subject
                ),
                "k": k,
                "so_dia_phuong": len(data),
                "silhouette_mean": (
                    run_df["silhouette"].mean()
                ),
                "silhouette_std": (
                    run_df["silhouette"].std()
                ),
                "silhouette_min": (
                    run_df["silhouette"].min()
                ),
                "silhouette_max": (
                    run_df["silhouette"].max()
                ),
                "ari_mean": (
                    np.mean(ari_values)
                ),
                "ari_min": (
                    np.min(ari_values)
                ),
                "valid_runs": int(
                    run_df["cum_hop_le"].sum()
                ),
                "total_runs": len(run_df)
            })

    finally:
        spark.stop()

    if not all_runs:
        print("KHONG CO MON NAO DU DIEU KIEN.")
        return

    # ================================================
    # 8. LUU KET QUA
    # ================================================

    runs = pd.concat(
        all_runs,
        ignore_index=True
    )

    summary = pd.DataFrame(summaries)

    runs_file = (
        OUTPUT / "kmeans_stability_runs.csv"
    )

    summary_file = (
        OUTPUT / "kmeans_stability_summary.csv"
    )

    runs.to_csv(
        runs_file,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    # ================================================
    # 9. HIEN THI TONG HOP
    # ================================================

    print("\n" + "=" * 75)
    print("TONG HOP DO ON DINH K-MEANS")
    print("=" * 75)

    print(
        summary[
            [
                "ten_mon",
                "k",
                "so_dia_phuong",
                "silhouette_mean",
                "silhouette_std",
                "ari_mean",
                "ari_min",
                "valid_runs"
            ]
        ].round(4).to_string(index=False)
    )

    print("\nDa luu:")
    print(runs_file)
    print(summary_file)

    print("\nHOAN THANH KIEM TRA K-MEANS!")


if __name__ == "__main__":
    main()
