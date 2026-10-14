
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import pandas as pd


# =====================================================
# 1. CAU HINH MOI TRUONG WINDOWS
# =====================================================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

HADOOP_DIR = Path(r"C:\hadoop")

if (HADOOP_DIR / "bin").is_dir():
    os.environ["HADOOP_HOME"] = str(HADOOP_DIR)
    os.environ["PATH"] = (
        str(HADOOP_DIR / "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# =====================================================
# 2. CAU HINH THUC NGHIEM
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "thpt_2026_clean.parquet"
)

OUTPUT = (
    ROOT
    / "data"
    / "analysis"
    / "benchmark"
)

REPEATS = 5

FRACTIONS = (0.25, 0.50, 1.00)

SPARK_MASTERS = ("local[2]", "local[4]")

SEED = 2026


# =====================================================
# 3. PANDAS - TINH DIEM TOAN THEO TINH
# =====================================================

def process_pandas(df):

    df = df.copy()

    df["province_code"] = (
        df["province_code"]
        .astype("string")
        .str.zfill(2)
    )

    df["toan"] = pd.to_numeric(
        df["toan"],
        errors="coerce"
    )

    df = df[
        df["toan"].between(0, 10)
    ]

    result = (
        df.groupby(
            "province_code",
            as_index=False
        )
        .agg(
            count=("toan", "count"),
            mean=("toan", "mean")
        )
        .sort_values("province_code")
        .reset_index(drop=True)
    )

    return result


# =====================================================
# 4. PYSPARK - TINH DIEM TOAN THEO TINH
# =====================================================

def process_spark(sdf):

    result = (
        sdf.withColumn(
            "province_code",
            F.lpad(
                F.col("province_code").cast("string"),
                2,
                "0"
            )
        )
        .withColumn(
            "toan",
            F.col("toan").cast("double")
        )
        .filter(
            F.col("toan").between(0.0, 10.0)
        )
        .groupBy("province_code")
        .agg(
            F.count("*").alias("count"),
            F.avg("toan").alias("mean")
        )
        .orderBy("province_code")
    )

    return result


# =====================================================
# 5. KIEM TRA KET QUA PANDAS VA SPARK
# =====================================================

def validate_results(pandas_df, spark_df, fraction):

    pandas_df = (
        pandas_df.sort_values("province_code")
        .reset_index(drop=True)
    )

    spark_df = (
        spark_df.sort_values("province_code")
        .reset_index(drop=True)
    )

    merged = pandas_df.merge(
        spark_df,
        on="province_code",
        suffixes=("_pandas", "_spark"),
        how="outer",
        indicator=True,
        validate="one_to_one"
    )

    all_provinces_match = (
        merged["_merge"] == "both"
    ).all()

    counts_match = (
        merged["count_pandas"]
        == merged["count_spark"]
    ).all()

    if not all_provinces_match or not counts_match:
        raise RuntimeError(
            f"Ket qua khong khop tai mau {fraction:.0%}"
        )

    max_diff = (
        merged["mean_pandas"]
        - merged["mean_spark"]
    ).abs().max()

    if pd.notna(max_diff) and max_diff > 1e-9:
        raise RuntimeError(
            f"Sai lech diem trung binh qua lon: {max_diff}"
        )

    return float(max_diff) if pd.notna(max_diff) else 0.0


# =====================================================
# 6. TAO CAC TAP DU LIEU MAU
# =====================================================

def prepare_datasets():

    print("\n=== DANG TAO CAC TAP DU LIEU ===")

    all_rows = pd.read_parquet(
        INPUT,
        columns=["province_code", "toan"]
    )

    all_rows["province_code"] = (
        all_rows["province_code"]
        .astype("string")
        .str.zfill(2)
    )

    datasets = {}
    sample_sizes = {}

    for fraction in FRACTIONS:

        if fraction == 1.0:
            sample = all_rows
        else:
            sample = all_rows.sample(
                frac=fraction,
                random_state=SEED
            ).reset_index(drop=True)

        filename = (
            OUTPUT
            / f"sample_{int(fraction * 100):03d}.parquet"
        )

        sample.to_parquet(
            filename,
            index=False
        )

        datasets[fraction] = filename
        sample_sizes[fraction] = len(sample)

        print(
            f"Mau {fraction:.0%}: "
            f"{len(sample):,} ban ghi"
        )

    del all_rows
    gc.collect()

    return datasets, sample_sizes


# =====================================================
# 7. THUC NGHIEM PANDAS
# =====================================================

def benchmark_pandas(datasets, records):

    reference = {}

    for fraction, filename in datasets.items():

        print(
            f"\n========== PANDAS {fraction:.0%} =========="
        )

        for run in range(1, REPEATS + 1):

            gc.collect()

            start = time.perf_counter()

            df = pd.read_parquet(
                filename,
                columns=["province_code", "toan"]
            )

            result = process_pandas(df)

            elapsed = time.perf_counter() - start

            reference[fraction] = result.copy()

            records.append({
                "fraction": fraction,
                "rows": len(df),
                "engine": "Pandas",
                "run": run,
                "seconds": elapsed,
                "province_count": len(result)
            })

            print(
                f"Lan {run}: {elapsed:.3f} giay"
            )

            del df, result

    return reference


# =====================================================
# 8. THUC NGHIEM PYSPARK
# =====================================================

def benchmark_spark(
    datasets,
    sample_sizes,
    reference,
    records
):

    for master in SPARK_MASTERS:

        print(
            f"\n========== PYSPARK {master} =========="
        )

        init_start = time.perf_counter()

        spark = (
            SparkSession.builder
            .appName(f"THPT_Benchmark_{master}")
            .master(master)
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

        init_seconds = (
            time.perf_counter() - init_start
        )

        print(
            f"Khoi tao Spark: {init_seconds:.3f} giay"
        )
        print(
            "(Khong tinh thoi gian khoi tao vao benchmark)"
        )

        try:

            for fraction, filename in datasets.items():

                print(
                    f"\n--- Mau {fraction:.0%} ---"
                )

                for run in range(1, REPEATS + 1):

                    start = time.perf_counter()

                    sdf = (
                        spark.read.parquet(str(filename))
                        .select("province_code", "toan")
                    )

                    rows = process_spark(sdf).collect()

                    elapsed = (
                        time.perf_counter() - start
                    )

                    spark_result = pd.DataFrame(
                        [row.asDict() for row in rows],
                        columns=[
                            "province_code",
                            "count",
                            "mean"
                        ]
                    )

                    max_diff = validate_results(
                        reference[fraction],
                        spark_result,
                        fraction
                    )

                    records.append({
                        "fraction": fraction,
                        "rows": sample_sizes[fraction],
                        "engine": f"PySpark {master}",
                        "run": run,
                        "seconds": elapsed,
                        "province_count": len(rows)
                    })

                    print(
                        f"Lan {run}: {elapsed:.3f} giay "
                        f"| Sai lech: {max_diff:.2e}"
                    )

        finally:
            spark.stop()


# =====================================================
# 9. TONG HOP VA LUU KET QUA
# =====================================================

def save_results(records):

    raw = pd.DataFrame(records)

    raw_file = (
        OUTPUT / "benchmark_runs.csv"
    )

    raw.to_csv(
        raw_file,
        index=False,
        encoding="utf-8-sig"
    )

    summary = (
        raw.groupby(
            ["fraction", "rows", "engine"]
        )["seconds"]
        .agg(
            median_seconds="median",
            mean_seconds="mean",
            std_seconds="std",
            min_seconds="min",
            max_seconds="max"
        )
        .reset_index()
    )

    summary_file = (
        OUTPUT / "benchmark_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 65)
    print("KET QUA THUC NGHIEM")
    print("=" * 65)

    print(
        summary[
            [
                "fraction",
                "rows",
                "engine",
                "median_seconds",
                "mean_seconds"
            ]
        ].to_string(index=False)
    )

    print("\nDa luu:")
    print(raw_file)
    print(summary_file)


# =====================================================
# 10. MAIN
# =====================================================

def main():

    print("=" * 65)
    print("THUC NGHIEM HIEU NANG PANDAS VS PYSPARK")
    print("=" * 65)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Khong tim thay file: {INPUT}"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nDu lieu:", INPUT)
    print("So lan lap:", REPEATS)
    print("Ty le mau:", FRACTIONS)
    print("Spark:", SPARK_MASTERS)

    datasets, sample_sizes = prepare_datasets()

    records = []

    reference = benchmark_pandas(
        datasets,
        records
    )

    benchmark_spark(
        datasets,
        sample_sizes,
        reference,
        records
    )

    save_results(records)

    print("\nHOAN THANH THUC NGHIEM!")


if __name__ == "__main__":
    main()
