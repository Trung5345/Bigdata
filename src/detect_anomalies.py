
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

# ==========================================
# 1. DOC KET QUA PHAN TICH TU PYSPARK
# ==========================================

INPUT_DIR = "data/analysis"

df = pd.read_csv(
    f"{INPUT_DIR}/math_province_distribution.csv",
    dtype={"province_code": str}
)

stats = pd.read_csv(
    f"{INPUT_DIR}/math_province_stats.csv",
    dtype={"province_code": str}
)

# ==========================================
# 2. TAO MA TRAN PHO DIEM
# ==========================================

# Moi dong la 1 tinh, moi cot la 1 khoang diem
matrix = df.pivot_table(
    index=["province_code", "province_name"],
    columns="khoang_diem",
    values="so_bai",
    aggfunc="sum",
    fill_value=0
)

# Dam bao du 10 khoang diem
matrix = matrix.reindex(
    columns=range(10),
    fill_value=0
)

matrix = matrix.astype(float)

# Pho diem cua toan quoc
national_counts = matrix.sum(axis=0).to_numpy()

results = []

# ==========================================
# 3. TINH JENSEN-SHANNON DIVERGENCE
# ==========================================

for (code, name), row in matrix.iterrows():

    province_counts = row.to_numpy()
    n = province_counts.sum()

    # So sanh voi phan con lai, khong gom tinh dang xet
    other_counts = national_counts - province_counts

    if n == 0 or other_counts.sum() == 0:
        continue

    # Chuyen tan suat thanh xac suat
    p = province_counts / n
    q = other_counts / other_counts.sum()

    # SciPy tra ve JS distance, nen binh phuong
    # de tinh Jensen-Shannon Divergence
    jsd = jensenshannon(p, q, base=2.0) ** 2

    # Do khac biet tong bien thien
    tvd = 0.5 * np.abs(p - q).sum()

    results.append({
        "province_code": code,
        "province_name": name,
        "so_bai": int(n),
        "js_divergence": float(jsd),
        "total_variation": float(tvd)
    })

# ==========================================
# 4. KET HOP THONG KE DIEM TRUNG BINH
# ==========================================

result_df = pd.DataFrame(results)

result_df = result_df.merge(
    stats[[
        "province_code",
        "diem_tb",
        "do_lech_chuan",
        "ty_le_duoi_5",
        "ty_le_tu_8"
    ]],
    on="province_code",
    how="left"
)

# Xep hang tu khac biet nhieu den it
result_df = result_df.sort_values(
    "js_divergence",
    ascending=False
).reset_index(drop=True)

result_df.insert(
    0,
    "hang_khac_biet",
    range(1, len(result_df) + 1)
)

# ==========================================
# 5. HIEN THI KET QUA
# ==========================================

print("\n=== XEP HANG KHAC BIET PHO DIEM TOAN ===")

print(
    result_df[[
        "hang_khac_biet",
        "province_name",
        "so_bai",
        "diem_tb",
        "js_divergence",
        "total_variation"
    ]].head(15).to_string(index=False)
)

print("\nLuu y: JSD cao khong dong nghia gian lan.")
print("Day la xep hang muc do khac biet thong ke.")

# ==========================================
# 6. LUU KET QUA
# ==========================================

output_file = (
    f"{INPUT_DIR}/math_anomaly_results.csv"
)

result_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nDA HOAN THANH TINH JSD!")
print("Ket qua:", os.path.abspath(output_file))
