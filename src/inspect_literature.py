
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

INPUT = "data/analysis/all_subjects/all_province_distribution.csv"
OUTPUT = "data/analysis/charts"
os.makedirs(OUTPUT, exist_ok=True)

df = pd.read_csv(INPUT, dtype={"province_code": str})
df["province_code"] = df["province_code"].str.zfill(2)

# Lay du lieu Ngu van
df = df[df["mon_thi"] == "ngu_van"]

matrix = df.pivot_table(
    index=["province_code", "province_name"],
    columns="khoang_diem",
    values="so_bai",
    aggfunc="sum",
    fill_value=0
).reindex(columns=range(10), fill_value=0)

national = matrix.sum(axis=0)

# Kiem tra 4 dia phuong
provinces = {
    "86": "Vinh Long",
    "80": "Tay Ninh",
    "14": "Son La",
    "31": "Hai Phong"
}

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(10)

for code, name in provinces.items():
    matches = [
        idx for idx in matrix.index if idx[0] == code
    ]

    if not matches:
        print("Khong tim thay:", name)
        continue

    counts = matrix.loc[matches[0]]
    other = national - counts

    p = counts.to_numpy(dtype=float)
    p = p / p.sum()

    q = other.to_numpy(dtype=float)
    q = q / q.sum()

    jsd = jensenshannon(p, q, base=2) ** 2

    print(f"\n{name}: JSD = {jsd:.6f}")

    print(
        pd.DataFrame({
            "khoang_diem": range(10),
            "ty_le_tinh": np.round(p * 100, 2),
            "ty_le_con_lai": np.round(q * 100, 2)
        }).to_string(index=False)
    )

    ax.plot(
        x,
        p * 100,
        marker="o",
        label=name
    )

national_pct = national / national.sum() * 100

ax.plot(
    x,
    national_pct,
    linestyle="--",
    marker="s",
    label="Toan quoc"
)

ax.set_xticks(x)
ax.set_xticklabels([
    "0-<1", "1-<2", "2-<3", "3-<4", "4-<5",
    "5-<6", "6-<7", "7-<8", "8-<9", "9-10"
])

ax.set_title("So sanh pho diem Ngu van THPT 2026")
ax.set_xlabel("Khoang diem")
ax.set_ylabel("Ty le bai thi (%)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

path = f"{OUTPUT}/literature_comparison.png"
plt.savefig(path, dpi=180)

print("\nDa luu:", path)

plt.show()
