
import os
import sys

# =========================================
# 1. CAU HINH SPARK TREN WINDOWS
# =========================================

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
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = (
    SparkSession.builder
    .appName("THPT_2026_KMeans")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.python.worker.reuse", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# =========================================
# 2. DOC DU LIEU THONG KE THEO TINH
# =========================================

stats = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("data/analysis/math_province_stats.csv")
)

dist = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("data/analysis/math_province_distribution.csv")
)

# =========================================
# 3. TAO DAC TRUNG PHO DIEM
# =========================================

# Chuyen so luong bai thi thanh ty le
province_totals = (
    dist.groupBy("province_code")
    .agg(F.sum("so_bai").alias("total"))
)

dist = dist.join(province_totals, "province_code")

dist = dist.withColumn(
    "ty_le",
    F.col("so_bai") / F.col("total")
)

# Tao 10 cot ty le: bin_0 ... bin_9
dist_pivot = (
    dist.groupBy("province_code")
    .pivot("khoang_diem", list(range(10)))
    .agg(F.first("ty_le"))
    .fillna(0)
)

for i in range(10):
    dist_pivot = dist_pivot.withColumnRenamed(
        str(i), f"bin_{i}"
    )

# Ghep voi thong ke trung binh
df = stats.join(
    dist_pivot,
    on="province_code",
    how="inner"
)

# Dac trung su dung cho K-means:
# Chi dung 10 ty le pho diem de tranh
# dem lap diem trung binh va ty le diem cao.
feature_cols = [f"bin_{i}" for i in range(10)]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

df = assembler.transform(df)

# =========================================
# 4. CHUAN HOA DAC TRUNG
# =========================================

# Cac bin da la ty le 0-1.
# StandardScaler giup cac bin co do bien thien
# khac nhau duoc dua ve cung thang do.
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(df)
df = scaler_model.transform(df).cache()

# =========================================
# 5. THU NGHIEM CAC GIA TRI K
# =========================================

evaluator = ClusteringEvaluator(
    featuresCol="scaled_features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

evaluations = []
models = {}

for k in range(2, 7):

    model = KMeans(
        k=k,
        seed=2026,
        featuresCol="scaled_features",
        predictionCol="prediction",
        maxIter=100
    ).fit(df)

    predictions = model.transform(df)

    score = evaluator.evaluate(predictions)

    evaluations.append((k, score))
    models[k] = model

    print(f"K={k}: Silhouette={score:.4f}")

# =========================================
# 6. CHON K TOT NHAT
# =========================================

best_k, best_score = max(
    evaluations,
    key=lambda item: item[1]
)

print("\nK duoc chon:", best_k)
print("Silhouette:", round(best_score, 4))

best_model = models[best_k]

predictions = best_model.transform(df)

# =========================================
# 7. TINH KHOANG CACH DEN TAM CUM
# =========================================

centers = best_model.clusterCenters()

# Chuyen vector thanh array bang Spark ML
from pyspark.ml.functions import vector_to_array

predictions = predictions.withColumn(
    "feature_array",
    vector_to_array("scaled_features")
)

# Tam cum duoc chon theo prediction
center_arrays = [
    [float(v) for v in center]
    for center in centers
]

# Chi co 34 tinh, broadcast mang tam cum nho
from pyspark.sql.types import DoubleType

def distance_to_center(values, cluster_id):
    center = center_arrays[int(cluster_id)]
    return sum(
        (float(a) - float(b)) ** 2
        for a, b in zip(values, center)
    ) ** 0.5

distance_udf = F.udf(
    distance_to_center,
    DoubleType()
)

predictions = predictions.withColumn(
    "distance_to_center",
    distance_udf(
        F.col("feature_array"),
        F.col("prediction")
    )
)

# =========================================
# 8. HIEN THI KET QUA
# =========================================

print("\n=== KET QUA PHAN CUM 34 TINH ===")

result = predictions.select(
    "province_code",
    "province_name",
    "so_bai",
    "diem_tb",
    "prediction",
    F.round(
        "distance_to_center", 4
    ).alias("distance_to_center")
).orderBy(
    F.col("distance_to_center").desc()
)

result.show(34, truncate=False)

print("\nSO TINH TRONG TUNG CUM:")

predictions.groupBy("prediction").count() \
    .orderBy("prediction") \
    .show()

# =========================================
# 9. XUAT KET QUA
# =========================================

output_dir = "data/analysis"
os.makedirs(output_dir, exist_ok=True)

result.toPandas().to_csv(
    f"{output_dir}/math_kmeans_results.csv",
    index=False,
    encoding="utf-8-sig"
)

import pandas as pd

pd.DataFrame(
    evaluations,
    columns=["k", "silhouette"]
).to_csv(
    f"{output_dir}/math_kmeans_evaluation.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nDA HOAN THANH K-MEANS!")
print("K tot nhat:", best_k)
print("Da luu ket qua tai:", output_dir)

df.unpersist()
spark.stop()
