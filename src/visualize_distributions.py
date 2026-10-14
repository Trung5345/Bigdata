
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# 1. CAU HINH
# =========================================

INPUT_FILE = "data/analysis/math_province_distribution.csv"
OUTPUT_DIR = "data/analysis/charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Doc du lieu tu ket qua PySpark
df = pd.read_csv(
    INPUT_FILE,
    dtype={"province_code": str}
)

# Chuan hoa ma tinh
df["province_code"] = df["province_code"].str.zfill(2)

# Cac tinh can so sanh
PROVINCES = {
    "14": "Son La",
    "11": "Dien Bien",
    "79": "Ho Chi Minh"
}

# =========================================
# 2. TAO BANG TAN SUAT
# =========================================

matrix = df.pivot_table(
    index="province_code",
    columns="khoang_diem",
    values="so_bai",
    aggfunc="sum",
    fill_value=0
).reindex(columns=range(10), fill_value=0)

# Phan bo toan quoc
national_counts = matrix.sum(axis=0)

# =========================================
# 3. VE BIEU DO
# =========================================

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(10)

for code, name in PROVINCES.items():
    if code not in matrix.index:
        print(f"Khong tim thay ma tinh {code}")
        continue

    counts = matrix.loc[code]
    other_counts = national_counts - counts

    province_pct = counts / counts.sum() * 100
    other_pct = other_counts / other_counts.sum() * 100

    # Duong phan bo cua tinh
    ax.plot(
        x,
        province_pct,
        marker="o",
        linewidth=2,
        label=name
    )

# Phan bo toan quoc (tham chieu chung)
national_pct = national_counts / national_counts.sum() * 100

ax.plot(
    x,
    national_pct,
    marker="s",
    linestyle="--",
    linewidth=2,
    label="Toan quoc",
)

ax.set_xticks(x)
ax.set_xticklabels(
    ["0-<1", "1-<2", "2-<3", "3-<4", "4-<5",
     "5-<6", "6-<7", "7-<8", "8-<9", "9-10"]
)

ax.set_title("So sanh pho diem Toan THPT 2026")
ax.set_xlabel("Khoang diem")
ax.set_ylabel("Ty le bai thi (%)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()

# Luu hinh
output_path = f"{OUTPUT_DIR}/math_province_comparison.png"
plt.savefig(output_path, dpi=180)
plt.show()

print("Da luu bieu do tai:", output_path)
