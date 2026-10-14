
from pathlib import Path
from itertools import combinations
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


# ============================================================
# 1. CAU HINH WINDOWS
# ============================================================

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
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# ============================================================
# 2. CAU HINH THUC NGHIEM
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT / "data" / "analysis" / "all_subjects"
)

OUTPUT_DIR = (
    ROOT / "data" / "analysis" / "kmeans_final"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISTRIBUTION_FILE = (
    DATA_DIR / "all_province_distribution.csv"
)

STATS_FILE = (
    DATA_DIR / "all_province_stats.csv"
)

SEEDS = [11, 22, 33, 44, 55]
K_VALUES = [2, 3, 4, 5, 6]

MIN_SAMPLES = 500
MIN_CLUSTER_SIZE = 3
MIN_VALID_SEEDS = 3
STABILITY_THRESHOLD = 0.80

FEATURE_COLS = [
    f"bin_{i}" for i in range(10)
]

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


# ============================================================
# 3. DOC DU LIEU VA TAO DAC TRUNG
# ============================================================

def load_features():

    print("\n=== DOC DU LIEU ===")

    for path in [
        DISTRIBUTION_FILE,
        STATS_FILE
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Khong tim thay: {path}"
            )

    dist = pd.read_csv(
        DISTRIBUTION_FILE,
        dtype={"province_code": str},
        encoding="utf-8-sig"
    )

    stats = pd.read_csv(
        STATS_FILE,
        dtype={"province_code": str},
        encoding="utf-8-sig"
    )

    for frame in [dist, stats]:
        frame["province_code"] = (
            frame["province_code"]
            .astype(str)
            .str.zfill(2)
        )

    # Mot dong la mot cap tinh - mon thi.
    # Muoi cot la tan suat diem trong 10 khoang.
    pivot = dist.pivot_table(
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

    # Chi lay dia phuong co tu 500 bai.
    pivot = pivot[
        pivot["so_bai"] >= MIN_SAMPLES
    ].copy()

    for col in FEATURE_COLS:
        pivot[col] = (
            pivot[col].astype(float)
            / pivot["so_bai"]
        )

    stats_small = (
        stats[
            [
                "mon_thi",
                "province_code",
                "diem_tb"
            ]
        ]
        .drop_duplicates(
            ["mon_thi", "province_code"]
        )
    )

    pivot = pivot.merge(
        stats_small,
        on=[
            "mon_thi",
            "province_code"
        ],
        how="left",
        validate="one_to_one"
    )

    if pivot["diem_tb"].isna().any():
        raise ValueError(
            "Thieu diem trung binh cua mot so dia phuong."
        )

    print(
        "So mon:",
        pivot["mon_thi"].nunique()
    )

    print(
        "So cap tinh-mon:",
        len(pivot)
    )

    return pivot


# ============================================================
# 4. TAO SPARK SESSION
# ============================================================

def create_spark():

    spark = (
        SparkSession.builder
        .appName("THPT_Final_KMeans")
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


# ============================================================
# 5. VECTOR HOA VA CHUAN HOA DAC TRUNG
# ============================================================

def prepare_subject_spark(spark, subject_df):

    records = []

    for row in subject_df.itertuples(index=False):

        record = {
            "province_code": str(
                row.province_code
            )
        }

        for col in FEATURE_COLS:
            record[col] = float(
                getattr(row, col)
            )

        records.append(record)

    sdf = spark.createDataFrame(records)

    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="raw_features"
    )

    sdf = assembler.transform(sdf)

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True
    )

    scaler_model = scaler.fit(sdf)

    sdf = scaler_model.transform(sdf)

    sdf = sdf.select(
        "province_code",
        "features"
    ).cache()

    sdf.count()

    return sdf


# ============================================================
# 6. HUAN LUYEN MOT MO HINH K-MEANS
# ============================================================

def run_kmeans(sdf, k, seed, evaluator):

    model = KMeans(
        k=k,
        seed=seed,
        featuresCol="features",
        predictionCol="prediction",
        maxIter=100,
        tol=1e-5
    ).fit(sdf)

    predictions = model.transform(sdf)

    rows = (
        predictions
        .select(
            "province_code",
            "features",
            "prediction"
        )
        .orderBy("province_code")
        .collect()
    )

    labels = np.asarray(
        [
            int(row["prediction"])
            for row in rows
        ],
        dtype=int
    )

    cluster_sizes = np.bincount(
        labels,
        minlength=k
    )

    valid = bool(
        len(rows) >= k * MIN_CLUSTER_SIZE
        and np.all(
            cluster_sizes >= MIN_CLUSTER_SIZE
        )
    )

    if len(np.unique(labels)) >= 2:
        silhouette = float(
            evaluator.evaluate(predictions)
        )
    else:
        silhouette = float("nan")

    # SUA LOI:
    # Spark MLlib clusterCenters() da tra ve
    # numpy.ndarray, khong duoc goi .toArray().
    centers = np.asarray(
        model.clusterCenters(),
        dtype=float
    )

    # Cot features la Spark ML Vector.
    features = np.asarray(
        [
            row["features"].toArray()
            for row in rows
        ],
        dtype=float
    )

    distances = np.linalg.norm(
        features - centers[labels],
        axis=1
    )

    assignments = pd.DataFrame({
        "province_code": [
            str(row["province_code"])
            for row in rows
        ],
        "cluster": labels,
        "distance_to_center": distances
    })

    return {
        "seed": seed,
        "k": k,
        "silhouette": silhouette,
        "valid": valid,
        "min_cluster_size": int(
            cluster_sizes.min()
        ),
        "cluster_sizes": str(
            cluster_sizes.tolist()
        ),
        "labels": labels,
        "assignments": assignments
    }


# ============================================================
# 7. TINH ARI GIUA CAC LAN CHAY
# ============================================================

def calculate_ari(runs):

    ari_values = []

    ari_by_seed = {
        run["seed"]: []
        for run in runs
    }

    for run_a, run_b in combinations(
        runs, 2
    ):

        ari = float(
            adjusted_rand_score(
                run_a["labels"],
                run_b["labels"]
            )
        )

        ari_values.append(ari)

        ari_by_seed[
            run_a["seed"]
        ].append(ari)

        ari_by_seed[
            run_b["seed"]
        ].append(ari)

    ari_mean = (
        float(np.mean(ari_values))
        if ari_values
        else np.nan
    )

    ari_min = (
        float(np.min(ari_values))
        if ari_values
        else np.nan
    )

    seed_ari = {
        seed: float(np.mean(values))
        if values else 0.0
        for seed, values in ari_by_seed.items()
    }

    return ari_mean, ari_min, seed_ari


# ============================================================
# 8. DANH GIA K-MEANS CHO TUNG MON
# ============================================================

def evaluate_subject(
    spark,
    subject_df,
    subject
):

    subject_name = SUBJECT_NAMES.get(
        subject,
        subject
    )

    print("\n" + "=" * 70)
    print("MON:", subject_name)
    print("SO DIA PHUONG:", len(subject_df))
    print("=" * 70)

    if len(subject_df) < 2 * MIN_CLUSTER_SIZE:

        print(
            "BO QUA: Khong du dia phuong."
        )

        return [], None, None

    sdf = prepare_subject_spark(
        spark,
        subject_df
    )

    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="prediction",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )

    evaluation_rows = []
    candidates = []

    try:

        for k in K_VALUES:

            if len(subject_df) < (
                k * MIN_CLUSTER_SIZE
            ):
                print(
                    f"Bo qua K={k}: "
                    "khong du dia phuong."
                )
                continue

            runs = []

            for seed in SEEDS:

                try:

                    result = run_kmeans(
                        sdf,
                        k,
                        seed,
                        evaluator
                    )

                    runs.append(result)

                    print(
                        f"K={k} | "
                        f"Seed={seed} | "
                        f"Silhouette="
                        f"{result['silhouette']:.4f} | "
                        f"MinSize="
                        f"{result['min_cluster_size']} | "
                        f"Valid={result['valid']}"
                    )

                except Exception as exc:

                    print(
                        f"LOI K={k}, Seed={seed}: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    # Khong dung toan bo qua trinh
                    # chi vi mot lan train that bai.
                    continue

            valid_runs = [
                run
                for run in runs
                if run["valid"]
                and np.isfinite(
                    run["silhouette"]
                )
            ]

            if len(valid_runs) < MIN_VALID_SEEDS:

                print(
                    f"Loai K={k}: "
                    f"chi co {len(valid_runs)} "
                    "lan chay hop le."
                )

                continue

            ari_mean, ari_min, seed_ari = (
                calculate_ari(valid_runs)
            )

            sil_values = [
                run["silhouette"]
                for run in valid_runs
            ]

            mean_sil = float(
                np.mean(sil_values)
            )

            std_sil = float(
                np.std(sil_values, ddof=0)
            )

            stable = bool(
                ari_mean >= STABILITY_THRESHOLD
            )

            # Diem ket hop do nhom tu dinh nghia.
            # Khong phai cong thuc chuan bat buoc.
            stability_factor = (
                0.7
                + 0.3 * max(
                    0.0, ari_mean
                )
            )

            selection_score = (
                mean_sil * stability_factor
            )

            row = {
                "mon_thi": subject,
                "ten_mon": subject_name,
                "k": k,
                "num_provinces": (
                    len(subject_df)
                ),
                "silhouette": mean_sil,
                "silhouette_std": std_sil,
                "ari_mean": ari_mean,
                "ari_min": ari_min,
                "valid_runs": len(valid_runs),
                "total_runs": len(SEEDS),
                "stable": stable,
                "selection_score": (
                    selection_score
                )
            }

            evaluation_rows.append(row)

            candidates.append({
                "evaluation": row,
                "runs": valid_runs,
                "seed_ari": seed_ari
            })

            print(
                f"--> K={k}: "
                f"Silhouette TB={mean_sil:.4f}, "
                f"ARI TB={ari_mean:.4f}, "
                f"Stable={stable}"
            )

    finally:
        sdf.unpersist()

    if not candidates:

        print(
            "KHONG TIM DUOC MO HINH HOP LE."
        )

        return evaluation_rows, None, None

    # Uu tien K co ARI >= 0.8.
    stable_candidates = [
        candidate
        for candidate in candidates
        if candidate[
            "evaluation"
        ]["stable"]
    ]

    pool = (
        stable_candidates
        if stable_candidates
        else candidates
    )

    best = max(
        pool,
        key=lambda candidate: (
            candidate["evaluation"][
                "selection_score"
            ],
            -candidate[
                "evaluation"
            ]["k"]
        )
    )

    # Chon seed dai dien, on dinh
    # so voi cac seed con lai.
    best_run = max(
        best["runs"],
        key=lambda run: (
            best["seed_ari"][
                run["seed"]
            ],
            run["silhouette"],
            -run["seed"]
        )
    )

    best_eval = best[
        "evaluation"
    ].copy()

    best_eval["selected_seed"] = (
        best_run["seed"]
    )

    print("\n>>> MO HINH DUOC CHON")
    print("K:", best_eval["k"])
    print("Seed:", best_run["seed"])

    print(
        "Silhouette TB:",
        round(
            best_eval["silhouette"],
            4
        )
    )

    print(
        "ARI TB:",
        round(
            best_eval["ari_mean"],
            4
        )
    )

    # Gan lai thong tin dia phuong.
    result_df = (
        best_run["assignments"]
        .merge(
            subject_df[
                [
                    "province_code",
                    "province_name",
                    "so_bai",
                    "diem_tb"
                ]
            ],
            on="province_code",
            how="left",
            validate="one_to_one"
        )
    )

    result_df["mon_thi"] = subject
    result_df["best_k"] = (
        best_eval["k"]
    )
    result_df["selected_seed"] = (
        best_run["seed"]
    )
    result_df["silhouette"] = (
        best_run["silhouette"]
    )
    result_df["ari_mean"] = (
        best_eval["ari_mean"]
    )

    result_df = result_df[
        [
            "mon_thi",
            "province_code",
            "province_name",
            "so_bai",
            "diem_tb",
            "cluster",
            "distance_to_center",
            "best_k",
            "selected_seed",
            "silhouette",
            "ari_mean"
        ]
    ]

    return (
        evaluation_rows,
        best_eval,
        result_df
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CHOT MO HINH K-MEANS CUOI CUNG")
    print("=" * 70)

    print("K values:", K_VALUES)
    print("Seeds:", SEEDS)
    print("Min samples:", MIN_SAMPLES)
    print(
        "Min cluster size:",
        MIN_CLUSTER_SIZE
    )
    print(
        "Stable ARI >=",
        STABILITY_THRESHOLD
    )

    start = time.perf_counter()

    features = load_features()

    spark = create_spark()

    all_evaluations = []
    all_selected = []
    all_results = []

    try:

        for subject in sorted(
            features["mon_thi"].unique()
        ):

            subject_df = (
                features[
                    features["mon_thi"] == subject
                ]
                .sort_values(
                    "province_code"
                )
                .reset_index(drop=True)
            )

            (
                evaluation_rows,
                selected,
                results
            ) = evaluate_subject(
                spark,
                subject_df,
                subject
            )

            all_evaluations.extend(
                evaluation_rows
            )

            if selected is not None:
                all_selected.append(
                    selected
                )

            if results is not None:
                all_results.append(
                    results
                )

    finally:
        spark.stop()

    if not all_results:
        raise RuntimeError(
            "Khong co mo hinh hop le."
        )

    evaluation_df = pd.DataFrame(
        all_evaluations
    )

    selected_df = pd.DataFrame(
        all_selected
    )

    result_df = pd.concat(
        all_results,
        ignore_index=True
    )

    # ==========================================
    # 10. LUU FILE
    # ==========================================

    evaluation_df.to_csv(
        OUTPUT_DIR
        / "all_kmeans_evaluation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    selected_df.to_csv(
        OUTPUT_DIR
        / "kmeans_model_selection.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result_df.to_csv(
        OUTPUT_DIR
        / "all_kmeans_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    elapsed = (
        time.perf_counter() - start
    )

    print("\n" + "=" * 85)
    print("TONG HOP MO HINH CUOI CUNG")
    print("=" * 85)

    columns = [
        "ten_mon",
        "k",
        "selected_seed",
        "silhouette",
        "ari_mean",
        "stable"
    ]

    print(
        selected_df[columns]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nTong so mon:",
        len(selected_df)
    )

    print(
        "Tong so dong ket qua:",
        len(result_df)
    )

    print(
        "Thoi gian thuc hien:",
        round(elapsed, 2),
        "giay"
    )

    print("\nDa luu vao:")
    print(OUTPUT_DIR)

    print(
        "\nHOAN THANH CHOT MO HINH!"
    )


if __name__ == "__main__":
    main()
