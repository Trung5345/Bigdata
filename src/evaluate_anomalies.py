
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

# =============================================
# 1. CAU HINH
# =============================================

INPUT_DIR = "data/analysis"
OUTPUT_FILE = f"{INPUT_DIR}/math_anomaly_evaluation.csv"

# So lan mo phong cho moi tinh
N_SIMULATIONS = 5000

# Muc y nghia thong ke
ALPHA = 0.05

# Seed de tai lap ket qua
rng = np.random.default_rng(2026)

# =============================================
# 2. DOC DU LIEU
# =============================================

df = pd.read_csv(
    f"{INPUT_DIR}/math_province_distribution.csv",
    dtype={"province_code": str}
)

df["province_code"] = df["province_code"].str.zfill(2)

stats = pd.read_csv(
    f"{INPUT_DIR}/math_province_stats.csv",
    dtype={"province_code": str}
)

stats["province_code"] = (
    stats["province_code"].str.zfill(2)
)

# =============================================
# 3. TAO MA TRAN PHO DIEM
# =============================================

matrix = df.pivot_table(
    index=["province_code", "province_name"],
    columns="khoang_diem",
    values="so_bai",
    aggfunc="sum",
    fill_value=0
).reindex(columns=range(10), fill_value=0)

matrix = matrix.astype(int)

national_counts = matrix.sum(axis=0).to_numpy()

results = []

# =============================================
# 4. MO PHONG MONTE CARLO
# =============================================

for (code, name), row in matrix.iterrows():

    province_counts = row.to_numpy()
    n = int(province_counts.sum())

    other_counts = national_counts - province_counts

    if n == 0 or other_counts.sum() == 0:
        continue

    # Phan bo thuc te cua tinh
    p = province_counts / n

    # Phan bo tham chieu (33 tinh con lai)
    q = other_counts / other_counts.sum()

    # JSD thuc te
    observed_jsd = (
        jensenshannon(p, q, base=2.0) ** 2
    )

    # Mo phong theo phan bo tham chieu
    simulated_counts = rng.multinomial(
        n=n,
        pvals=q,
        size=N_SIMULATIONS
    )

    simulated_probs = simulated_counts / n

    # Tinh JSD cho tung mau mo phong
    simulated_jsd = np.array([
        jensenshannon(sim_p, q, base=2.0) ** 2
        for sim_p in simulated_probs
    ])

    # Nguong JSD tai phan vi 95%
    threshold_95 = np.quantile(
        simulated_jsd, 0.95
    )

    # P-value mo phong
    p_value = (
        1 + np.sum(simulated_jsd >= observed_jsd)
    ) / (N_SIMULATIONS + 1)

    # Do lon khac biet
    tvd = 0.5 * np.abs(p - q).sum()

    results.append({
        "province_code": code,
        "province_name": name,
        "so_bai": n,
        "js_divergence": observed_jsd,
        "threshold_95": threshold_95,
        "p_value": p_value,
        "total_variation": tvd
    })

    print(
        f"{name}: "
        f"JSD={observed_jsd:.6f}, "
        f"threshold={threshold_95:.6f}, "
        f"p={p_value:.6f}"
    )

# =============================================
# 5. HIEU CHINH KIEM DINH NHIEU TINH
# =============================================

result_df = pd.DataFrame(results)

# Benjamini-Hochberg FDR correction
m = len(result_df)

result_df = result_df.sort_values(
    "p_value"
).reset_index(drop=True)

ranks = np.arange(1, m + 1)

adjusted = (
    result_df["p_value"].to_numpy()
    * m / ranks
)

adjusted = np.minimum.accumulate(
    adjusted[::-1]
)[::-1]

result_df["p_value_fdr"] = np.clip(
    adjusted, 0, 1
)

# =============================================
# 6. PHAN LOAI KET QUA
# =============================================

# Luu y: day la khac biet thong ke,
# khong phai ket luan gian lan

result_df["khac_biet_co_y_nghia"] = (
    result_df["p_value_fdr"] < ALPHA
)

result_df = result_df.sort_values(
    "js_divergence",
    ascending=False
).reset_index(drop=True)

result_df.insert(
    0,
    "hang_khac_biet",
    np.arange(1, len(result_df) + 1)
)

# Ghep them diem trung binh
result_df = result_df.merge(
    stats[[
        "province_code",
        "diem_tb",
        "ty_le_duoi_5",
        "ty_le_tu_8"
    ]],
    on="province_code",
    how="left"
)

# =============================================
# 7. HIEN THI KET QUA
# =============================================

print("\n=== DANH GIA KHAC BIET PHO DIEM ===")

print(
    result_df[[
        "hang_khac_biet",
        "province_name",
        "so_bai",
        "diem_tb",
        "js_divergence",
        "threshold_95",
        "p_value_fdr",
        "khac_biet_co_y_nghia"
    ]].to_string(index=False)
)

# =============================================
# 8. LUU KET QUA
# =============================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nDA HOAN THANH DANH GIA MONTE CARLO!")
print("Ket qua:", os.path.abspath(OUTPUT_FILE))
print("So lan mo phong moi tinh:", N_SIMULATIONS)
print("Muc y nghia FDR:", ALPHA)
