
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import false_discovery_control

# ==========================================
# 1. CAU HINH
# ==========================================

INPUT_DIR = "data/analysis/all_subjects"

DIST_FILE = f"{INPUT_DIR}/all_province_distribution.csv"
STATS_FILE = f"{INPUT_DIR}/all_province_stats.csv"
OUTPUT_FILE = f"{INPUT_DIR}/all_anomaly_evaluation.csv"

N_SIMULATIONS = 2000
MIN_SAMPLES = 100
ALPHA = 0.05

# Nguong do lon khac biet de sang loc
JSD_THRESHOLD = 0.02
TVD_THRESHOLD = 0.10

rng = np.random.default_rng(2026)

# ==========================================
# 2. DOC DU LIEU
# ==========================================

dist = pd.read_csv(
    DIST_FILE,
    dtype={"province_code": str}
)

stats = pd.read_csv(
    STATS_FILE,
    dtype={"province_code": str}
)

dist["province_code"] = dist["province_code"].str.zfill(2)
stats["province_code"] = stats["province_code"].str.zfill(2)

results = []

# ==========================================
# 3. PHAN TICH TUNG MON
# ==========================================

for subject in sorted(dist["mon_thi"].unique()):

    print(f"\nDang danh gia: {subject}")

    subject_df = dist[
        dist["mon_thi"] == subject
    ]

    matrix = subject_df.pivot_table(
        index=["province_code", "province_name"],
        columns="khoang_diem",
        values="so_bai",
        aggfunc="sum",
        fill_value=0
    ).reindex(columns=range(10), fill_value=0)

    national_counts = matrix.sum(axis=0).to_numpy(
        dtype=float
    )

    for (code, name), row in matrix.iterrows():

        counts = row.to_numpy(dtype=float)
        n = int(counts.sum())

        other_counts = national_counts - counts
        other_n = other_counts.sum()

        if n == 0 or other_n <= 0:
            continue

        p = counts / n
        q = other_counts / other_n

        jsd = float(
            jensenshannon(p, q, base=2.0) ** 2
        )

        tvd = float(
            0.5 * np.abs(p - q).sum()
        )

        p_value = np.nan
        threshold_95 = np.nan

        if n >= MIN_SAMPLES:

            simulated = rng.multinomial(
                n=n,
                pvals=q,
                size=N_SIMULATIONS
            )

            sim_jsd = np.array([
                jensenshannon(
                    counts_sim / n,
                    q,
                    base=2.0
                ) ** 2
                for counts_sim in simulated
            ])

            threshold_95 = float(
                np.quantile(sim_jsd, 0.95)
            )

            p_value = (
                1 + np.sum(sim_jsd >= jsd)
            ) / (N_SIMULATIONS + 1)

        results.append({
            "mon_thi": subject,
            "province_code": code,
            "province_name": name,
            "so_bai": n,
            "js_divergence": jsd,
            "total_variation": tvd,
            "threshold_95": threshold_95,
            "p_value": p_value
        })

# ==========================================
# 4. HIEU CHINH KIEM DINH NHIEU LAN
# ==========================================

result = pd.DataFrame(results)

result["p_value_fdr"] = np.nan

valid = result["p_value"].notna()

if valid.any():
    result.loc[valid, "p_value_fdr"] = (
        false_discovery_control(
            result.loc[valid, "p_value"].to_numpy(),
            method="bh"
        )
    )

# ==========================================
# 5. DANH GIA KET QUA
# ==========================================

result["du_mau"] = result["so_bai"] >= MIN_SAMPLES

result["co_y_nghia_thong_ke"] = (
    result["p_value_fdr"] < ALPHA
).fillna(False)

# Quy tac sang loc ban dau:
# Can du mau + co y nghia thong ke
# + do lon khac biet vuot nguong thuc nghiem

result["can_xem_xet"] = (
    result["du_mau"]
    & result["co_y_nghia_thong_ke"]
    & (result["js_divergence"] >= JSD_THRESHOLD)
    & (result["total_variation"] >= TVD_THRESHOLD)
)

# Ghep diem trung binh
result = result.merge(
    stats[[
        "mon_thi",
        "province_code",
        "diem_tb",
        "ty_le_duoi_5",
        "ty_le_tu_8"
    ]],
    on=["mon_thi", "province_code"],
    how="left"
)

result = result.sort_values(
    ["mon_thi", "js_divergence"],
    ascending=[True, False]
).reset_index(drop=True)

result["hang_trong_mon"] = (
    result.groupby("mon_thi").cumcount() + 1
)

# ==========================================
# 6. IN KET QUA TONG HOP
# ==========================================

print("\n=== TONG HOP DANH GIA 12 MON ===")

summary = result.groupby("mon_thi").agg(
    so_dia_phuong=("province_code", "count"),
    du_mau=("du_mau", "sum"),
    co_y_nghia=("co_y_nghia_thong_ke", "sum"),
    can_xem_xet=("can_xem_xet", "sum"),
    jsd_lon_nhat=("js_divergence", "max")
)

print(summary.to_string())

print("\n=== CAC TRUONG HOP CAN XEM XET ===")

flagged = result[result["can_xem_xet"]]

print(
    flagged[[
        "mon_thi",
        "province_name",
        "so_bai",
        "diem_tb",
        "js_divergence",
        "total_variation",
        "p_value_fdr"
    ]].to_string(index=False)
)

# ==========================================
# 7. LUU KET QUA
# ==========================================

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nDA HOAN THANH DANH GIA!")
print("Tong so cap mon-tinh:", len(result))
print("So truong hop can xem xet:", len(flagged))
print("File:", os.path.abspath(OUTPUT_FILE))
