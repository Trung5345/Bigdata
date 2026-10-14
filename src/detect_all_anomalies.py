
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

# ==========================================
# 1. CAU HINH
# ==========================================

INPUT_DIR = "data/analysis/all_subjects"

DISTRIBUTION_FILE = (
    f"{INPUT_DIR}/all_province_distribution.csv"
)

STATS_FILE = (
    f"{INPUT_DIR}/all_province_stats.csv"
)

OUTPUT_FILE = (
    f"{INPUT_DIR}/all_anomaly_results.csv"
)

# So bai toi thieu de duoc dua vao
# danh sach uu tien phan tich
MIN_SAMPLES = 100

# ==========================================
# 2. DOC DU LIEU
# ==========================================

distribution = pd.read_csv(
    DISTRIBUTION_FILE,
    dtype={"province_code": str}
)

stats = pd.read_csv(
    STATS_FILE,
    dtype={"province_code": str}
)

distribution["province_code"] = (
    distribution["province_code"].str.zfill(2)
)

stats["province_code"] = (
    stats["province_code"].str.zfill(2)
)

results = []

# ==========================================
# 3. PHAN TICH THEO TUNG MON
# ==========================================

subjects = sorted(
    distribution["mon_thi"].unique()
)

for subject in subjects:

    print(f"\nDang phan tich mon: {subject}")

    subject_df = distribution[
        distribution["mon_thi"] == subject
    ]

    # Ma tran 34 tinh x 10 khoang diem
    matrix = subject_df.pivot_table(
        index=["province_code", "province_name"],
        columns="khoang_diem",
        values="so_bai",
        aggfunc="sum",
        fill_value=0
    ).reindex(
        columns=range(10),
        fill_value=0
    )

    national_counts = (
        matrix.sum(axis=0).to_numpy(dtype=float)
    )

    # ======================================
    # 4. TINH JSD VA TVD TUNG TINH
    # ======================================

    for (code, name), row in matrix.iterrows():

        counts = row.to_numpy(dtype=float)
        n = int(counts.sum())

        # Phan bo 33 tinh con lai
        other_counts = national_counts - counts

        if n == 0 or other_counts.sum() <= 0:
            continue

        p = counts / n
        q = other_counts / other_counts.sum()

        # Jensen-Shannon Divergence
        jsd = float(
            jensenshannon(p, q, base=2.0) ** 2
        )

        # Total Variation Distance
        tvd = float(
            0.5 * np.abs(p - q).sum()
        )

        # Kiem tra quy mo mau
        enough_samples = n >= MIN_SAMPLES

        results.append({
            "mon_thi": subject,
            "province_code": code,
            "province_name": name,
            "so_bai": n,
            "js_divergence": jsd,
            "total_variation": tvd,
            "du_mau": enough_samples
        })

# ==========================================
# 5. KET HOP THONG KE
# ==========================================

result_df = pd.DataFrame(results)

result_df = result_df.merge(
    stats[[
        "mon_thi",
        "province_code",
        "diem_tb",
        "do_lech_chuan",
        "ty_le_duoi_5",
        "ty_le_tu_8"
    ]],
    on=["mon_thi", "province_code"],
    how="left"
)

# Xep hang JSD rieng cho tung mon
result_df = result_df.sort_values(
    ["mon_thi", "js_divergence"],
    ascending=[True, False]
).reset_index(drop=True)

result_df["hang_khac_biet"] = (
    result_df.groupby("mon_thi")
    .cumcount() + 1
)

# ==========================================
# 6. HIEN THI TOP 5 TUNG MON
# ==========================================

for subject in subjects:

    print("\n================================")
    print("MON:", subject.upper())
    print("================================")

    subset = result_df[
        (result_df["mon_thi"] == subject)
        & result_df["du_mau"]
    ]

    print(
        subset[[
            "province_name",
            "so_bai",
            "diem_tb",
            "js_divergence",
            "total_variation"
        ]].head(5).to_string(index=False)
    )

# ==========================================
# 7. XUAT KET QUA
# ==========================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n================================")
print("DA HOAN THANH PHAN TICH 12 MON!")
print("Tong so ket qua:", len(result_df))
print("File:", os.path.abspath(OUTPUT_FILE))
print("================================")
