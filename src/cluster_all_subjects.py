
import os
import sys
import math
import pandas as pd

# ==========================================
# 1. CAU HINH MOI TRUONG WINDOWS
# ==========================================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

HADOOP_HOME = r"C:\hadoop"

if os.path.isdir(os.path.join(HADOOP_HOME, "bin")):
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    os.environ["PATH"] = (
        os.path.join(HADOOP_HOME, "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# ==========================================
# 2. CAU HINH DU AN
# ==========================================

INPUT_DIR = "data/analysis/all_subjects"

DIST_FILE = (
    f"{INPUT_DIR}/all_province_distribution.csv"
)
STATS_FILE = (
    f"{INPUT_DIR}/all_province_stats.csv"
)

RESULT_FILE = (
    f"{INPUT_DIR}/all_kmeans_results.csv"
)
EVALUATION_FILE = (
    f"{INPUT_DIR}/all_kmeans_evaluation.csv"
)

MIN_SAMPLES = 500
MIN_PROVINCES = 5
MIN_CLUSTER_SIZE = 3

MIN_K = 2
MAX_K = 6

SEED = 2026

FEATURE_COLS = [
    f"bin_{i}" for i in range(10)
]

# ==========================================
# 3. KHOI TAO SPARK
# ==========================================

spark = (
    SparkSession.builder
    .appName("THPT_2026_All_KMeans")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.python.worker.reuse", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # ======================================
    # 4. DOC DU LIEU
    # ======================================

    print("\nDANG DOC DU LIEU...")

    dist = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(DIST_FILE)
    )

    stats = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(STATS_FILE)
    )

    # Giu ma tinh o dang chuoi 2 ky tu
    dist = dist.withColumn(
        "province_code",
        F.lpad(
            F.col("province_code").cast("string"),
            2,
            "0"
        )
    )

    stats = stats.withColumn(
        "province_code",
        F.lpad(
            F.col("province_code").cast("string"),
            2,
            "0"
        )
    )

    subjects = [
        row["mon_thi"]
        for row in (
            dist.select("mon_thi")
            .distinct()
            .orderBy("mon_thi")
            .collect()
        )
    ]

    print("Tong so mon:", len(subjects))

    all_results = []
    all_evaluations = []
    skipped_subjects = []

    # ======================================
    # 5. PHAN CUM TUNG MON
    # ======================================

    for subject in subjects:

        print("\n" + "=" * 50)
        print("DANG PHAN CUM:", subject.upper())
        print("=" * 50)

        subject_dist = dist.filter(
            F.col("mon_thi") == subject
        )

        # Tong so bai cua moi tinh
        totals = (
            subject_dist
            .groupBy("province_code")
            .agg(
                F.sum("so_bai").alias("total")
            )
        )

        # Chi giu tinh co tu 500 bai
        subject_dist = (
            subject_dist
            .join(totals, "province_code")
            .filter(
                F.col("total") >= MIN_SAMPLES
            )
            .withColumn(
                "ty_le",
                F.col("so_bai") / F.col("total")
            )
        )

        # ==================================
        # 6. TAO 10 DAC TRUNG PHO DIEM
        # ==================================

        pivot = (
            subject_dist
            .groupBy("province_code")
            .pivot(
                "khoang_diem",
                list(range(10))
            )
            .agg(F.first("ty_le"))
            .fillna(0)
        )

        for i in range(10):
            pivot = pivot.withColumnRenamed(
                str(i),
                f"bin_{i}"
            )

        subject_stats = stats.filter(
            F.col("mon_thi") == subject
        )

        df = subject_stats.join(
            pivot,
            "province_code",
            "inner"
        )

        n_provinces = df.count()

        print(
            "So dia phuong du mau:",
            n_provinces
        )

        if n_provinces < MIN_PROVINCES:
            print(
                "Bo qua: Khong du",
                MIN_PROVINCES,
                "dia phuong"
            )

            skipped_subjects.append(subject)
            continue

        # ==================================
        # 7. VECTOR HOA DAC TRUNG
        # ==================================

        assembler = VectorAssembler(
            inputCols=FEATURE_COLS,
            outputCol="features"
        )

        df = assembler.transform(df)

        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withMean=True,
            withStd=True
        )

        scaler_model = scaler.fit(df)

        df = (
            scaler_model
            .transform(df)
            .cache()
        )

        # ==================================
        # 8. DANH GIA SILHOUETTE
        # ==================================

        evaluator = ClusteringEvaluator(
            featuresCol="scaled_features",
            predictionCol="prediction",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean"
        )

        models = {}
        scores = []

        max_k = min(
            MAX_K,
            n_provinces - 1,
            n_provinces // MIN_CLUSTER_SIZE
        )

        for k in range(MIN_K, max_k + 1):

            model = KMeans(
                k=k,
                seed=SEED,
                maxIter=100,
                featuresCol="scaled_features",
                predictionCol="prediction"
            ).fit(df)

            pred = model.transform(df)

            cluster_sizes = [
                row["count"]
                for row in (
                    pred.groupBy("prediction")
                    .count()
                    .collect()
                )
            ]

            # Kiem tra du k cum
            if len(cluster_sizes) != k:
                print(
                    f"K={k}: bo qua vi khong du cum"
                )
                continue

            min_cluster_size = min(
                cluster_sizes
            )

            # Moi cum co it nhat 3 tinh
            if min_cluster_size < MIN_CLUSTER_SIZE:
                print(
                    f"K={k}: bo qua vi co cum "
                    f"duoi {MIN_CLUSTER_SIZE} tinh"
                )
                continue

            score = float(
                evaluator.evaluate(pred)
            )

            models[k] = model
            scores.append((k, score))

            all_evaluations.append({
                "mon_thi": subject,
                "k": k,
                "silhouette": score,
                "min_cluster_size": min_cluster_size,
                "so_dia_phuong": n_provinces
            })

            print(
                f"K={k}: "
                f"silhouette={score:.4f}, "
                f"cum_nho_nhat={min_cluster_size}"
            )

        # ==================================
        # 9. CHON K TOT NHAT
        # ==================================

        if not scores:
            print(
                "Khong co mo hinh hop le cho mon:",
                subject
            )

            skipped_subjects.append(subject)
            df.unpersist()
            continue

        best_k, best_score = max(
            scores,
            key=lambda item: item[1]
        )

        print("\nK duoc chon:", best_k)
        print(
            "Silhouette:",
            round(best_score, 4)
        )

        best_model = models[best_k]

        # ==================================
        # 10. TINH KHOANG CACH TAM CUM
        # ==================================

        centers = best_model.clusterCenters()

        center_values = [
            [float(value) for value in center]
            for center in centers
        ]

        predictions = best_model.transform(df)

        # So tinh it, thu thap vector ve driver
        # de tinh khoang cach ma khong can UDF.
        rows = predictions.select(
            "province_code",
            "province_name",
            "so_bai",
            "diem_tb",
            "prediction",
            "scaled_features"
        ).collect()

        for row in rows:

            cluster_id = int(
                row["prediction"]
            )

            features = [
                float(value)
                for value in row["scaled_features"]
            ]

            center = center_values[cluster_id]

            distance = math.sqrt(
                sum(
                    (a - b) ** 2
                    for a, b in zip(
                        features,
                        center
                    )
                )
            )

            all_results.append({
                "mon_thi": subject,
                "province_code": str(
                    row["province_code"]
                ).zfill(2),
                "province_name": row[
                    "province_name"
                ],
                "so_bai": int(
                    row["so_bai"]
                ),
                "diem_tb": float(
                    row["diem_tb"]
                ),
                "cluster": cluster_id,
                "distance_to_center": distance,
                "best_k": best_k,
                "silhouette": best_score
            })

        df.unpersist()

    # ======================================
    # 11. LUU KET QUA
    # ======================================

    results_df = pd.DataFrame(
        all_results
    )

    evaluation_df = pd.DataFrame(
        all_evaluations
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    evaluation_df.to_csv(
        EVALUATION_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================
    # 12. IN KET QUA TONG HOP
    # ======================================

    print("\n" + "=" * 50)
    print("KET QUA TONG HOP")
    print("=" * 50)

    if not results_df.empty:

        best_summary = (
            results_df[
                [
                    "mon_thi",
                    "best_k",
                    "silhouette"
                ]
            ]
            .drop_duplicates("mon_thi")
            .sort_values("mon_thi")
        )

        print(
            best_summary.to_string(
                index=False
            )
        )

    print("\nMon bo qua:", skipped_subjects)
    print("So ket qua:", len(results_df))
    print("File ket qua:", RESULT_FILE)
    print("File danh gia:", EVALUATION_FILE)

    print("\nDA HOAN THANH PHAN CUM!")

finally:
    spark.stop()
