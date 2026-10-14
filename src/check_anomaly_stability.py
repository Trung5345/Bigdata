
import os
import pandas as pd

INPUT = "data/analysis/all_subjects/all_anomaly_evaluation.csv"
OUTPUT_DIR = "data/analysis/all_subjects"

df = pd.read_csv(
    INPUT,
    dtype={"province_code": str}
)

df["province_code"] = df["province_code"].str.zfill(2)

# Cac cau hinh thu nghiem
configs = [
    (100, 0.02, 0.10),
    (300, 0.02, 0.10),
    (500, 0.02, 0.10),
    (1000, 0.02, 0.10),
    (500, 0.01, 0.10),
    (500, 0.03, 0.15)
]

results = []

for min_n, jsd_min, tvd_min in configs:

    flagged = df[
        (df["so_bai"] >= min_n)
        & (df["p_value_fdr"] < 0.05)
        & (df["js_divergence"] >= jsd_min)
        & (df["total_variation"] >= tvd_min)
    ]

    results.append({
        "min_samples": min_n,
        "jsd_threshold": jsd_min,
        "tvd_threshold": tvd_min,
        "so_truong_hop": len(flagged),
        "so_dia_phuong": flagged["province_code"].nunique()
    })

result_df = pd.DataFrame(results)

print("\n=== KIEM TRA DO ON DINH CANH BAO ===")
print(result_df.to_string(index=False))

result_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "anomaly_stability.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# Xem cac truong hop cua cau hinh n >= 500
selected = df[
    (df["so_bai"] >= 500)
    & (df["p_value_fdr"] < 0.05)
    & (df["js_divergence"] >= 0.02)
    & (df["total_variation"] >= 0.10)
]

print("\n=== SO CANH BAO THEO MON (N >= 500) ===")

print(
    selected.groupby("mon_thi")
    .size()
    .sort_values(ascending=False)
    .to_string()
)

print("\nDA HOAN THANH KIEM TRA!")
