
from pathlib import Path
import numpy as np
import pandas as pd

# ===================================================
# 1. DUONG DAN
# ===================================================

ROOT = Path(__file__).resolve().parents[1]

PARQUET = (
    ROOT / "data" / "processed"
    / "thpt_2026_clean.parquet"
)

ANOMALY = (
    ROOT / "data" / "analysis"
    / "all_subjects"
    / "all_anomaly_evaluation.csv"
)

PROVINCE_STATS = (
    ROOT / "data" / "analysis"
    / "all_subjects"
    / "all_province_stats.csv"
)

OUTPUT = (
    ROOT / "data" / "analysis"
    / "source_audit"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

# Khong tu dong coi cac cot nay la tuong duong
PAIRS = [
    ("gdcd", "gdkt_pl"),
    ("cong_nghe", "cong_nghe_nn"),
    ("cong_nghe", "cong_nghe_cn")
]

# ===================================================
# 2. KIEM TRA HAI COT CU
# ===================================================

def audit_columns():

    print("\n=== KIEM TRA GDCD VA CONG_NGHE ===")

    df = pd.read_parquet(PARQUET)

    cols = [
        "gdcd",
        "cong_nghe",
        "gdkt_pl",
        "cong_nghe_nn",
        "cong_nghe_cn"
    ]

    available = [
        col for col in cols
        if col in df.columns
    ]

    print("Tong ban ghi:", f"{len(df):,}")
    print("Cac cot:", available)

    for col in available:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    rows = []

    for col in available:

        values = df[col]

        valid = values.between(0, 10)

        rows.append({
            "cot": col,
            "co_gia_tri_so": int(
                values.notna().sum()
            ),
            "diem_hop_le": int(
                valid.sum()
            ),
            "diem_ngoai_0_10": int(
                (
                    values.notna()
                    & ~valid
                ).sum()
            ),
            "diem_trung_binh": (
                round(values[valid].mean(), 4)
                if valid.any()
                else np.nan
            )
        })

    result = pd.DataFrame(rows)

    print("\nSO LUONG GIA TRI:")
    print(result.to_string(index=False))

    result.to_csv(
        OUTPUT / "legacy_column_counts.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ===============================================
    # 3. KIEM TRA CAC CAP COT
    # ===============================================

    checks = []

    for old_col, new_col in PAIRS:

        if (
            old_col not in df.columns
            or new_col not in df.columns
        ):
            continue

        old = df[old_col]
        new = df[new_col]

        old_valid = old.between(0, 10)
        new_valid = new.between(0, 10)

        both = old_valid & new_valid

        equal = (
            both
            & np.isclose(
                old.fillna(-999).to_numpy(),
                new.fillna(-999).to_numpy(),
                atol=1e-8,
                rtol=0
            )
        )

        row = {
            "cot_cu": old_col,
            "cot_moi": new_col,

            "chi_cot_cu_co_diem": int(
                (old_valid & ~new_valid).sum()
            ),

            "chi_cot_moi_co_diem": int(
                (~old_valid & new_valid).sum()
            ),

            "ca_hai_co_diem": int(
                both.sum()
            ),

            "ca_hai_bang_nhau": int(
                equal.sum()
            ),

            "ca_hai_khac_nhau": int(
                (both & ~equal).sum()
            )
        }

        checks.append(row)

    comparison = pd.DataFrame(checks)

    print("\nDOI CHIEU CAC COT:")
    print(
        comparison.to_string(index=False)
    )

    comparison.to_csv(
        OUTPUT / "legacy_column_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ===============================================
    # 4. THONG KE THEO TINH
    # ===============================================

    if "province_name" in df.columns:

        grouped = df.groupby(
            "province_name",
            dropna=False
        )

        province_rows = []

        for province, group in grouped:

            row = {
                "province_name": province,
                "so_thi_sinh": len(group)
            }

            for col in available:
                row[f"{col}_co_diem"] = int(
                    group[col].between(0, 10).sum()
                )

            province_rows.append(row)

        province_df = pd.DataFrame(
            province_rows
        )

        province_df.to_csv(
            OUTPUT / "legacy_by_province.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # ===============================================
    # 5. GIA TRI CU CHUA CO O CAC COT MOI
    # ===============================================

    if "cong_nghe" in df.columns:

        old_valid = df["cong_nghe"].between(
            0, 10
        )

        nn_valid = (
            df["cong_nghe_nn"].between(0, 10)
            if "cong_nghe_nn" in df.columns
            else pd.Series(False, index=df.index)
        )

        cn_valid = (
            df["cong_nghe_cn"].between(0, 10)
            if "cong_nghe_cn" in df.columns
            else pd.Series(False, index=df.index)
        )

        not_classified = (
            old_valid
            & ~nn_valid
            & ~cn_valid
        )

        print(
            "\nCONG_NGHE CO DIEM NHUNG "
            "HAI COT MOI KHONG CO:",
            int(not_classified.sum())
        )

        unclassified_summary = (
            df.loc[not_classified]
            .groupby("province_name", dropna=False)
            .size()
            .reset_index(name="so_ban_ghi")
            .sort_values(
                "so_ban_ghi",
                ascending=False
            )
        )

        unclassified_summary.to_csv(
            OUTPUT / "cong_nghe_unclassified.csv",
            index=False,
            encoding="utf-8-sig"
        )

    if (
        "gdcd" in df.columns
        and "gdkt_pl" in df.columns
    ):

        gdcd_only = (
            df["gdcd"].between(0, 10)
            & ~df["gdkt_pl"].between(0, 10)
        )

        print(
            "GDCD CO DIEM NHUNG "
            "GDKT_PL KHONG CO:",
            int(gdcd_only.sum())
        )

    print("\nDA LUU BAO CAO TAI:")
    print(OUTPUT)


# ===================================================
# 6. XEP HANG NHUNG DIA PHUONG CO JSD CAO
# ===================================================

def audit_anomalies():

    print("\n=== DIA PHUONG CAN DOI CHIEU ===")

    if not ANOMALY.exists():
        print("Khong tim thay file anomaly.")
        return

    df = pd.read_csv(
        ANOMALY,
        dtype={"province_code": str}
    )

    df["province_code"] = (
        df["province_code"].str.zfill(2)
    )

    # Chi lay mau du 500 bai va cac nguong
    # da su dung trong do an
    selected = df[
        (df["so_bai"] >= 500)
        & (df["js_divergence"] >= 0.02)
        & (df["total_variation"] >= 0.10)
        & (df["p_value_fdr"] < 0.05)
    ].copy()

    selected = selected.sort_values(
        "js_divergence",
        ascending=False
    )

    # Ghep them diem TB neu co
    if PROVINCE_STATS.exists():

        stats = pd.read_csv(
            PROVINCE_STATS,
            dtype={"province_code": str}
        )

        stats["province_code"] = (
            stats["province_code"].str.zfill(2)
        )

        columns = [
            "mon_thi",
            "province_code",
            "diem_tb"
        ]

        if all(
            c in stats.columns
            for c in columns
        ):

            selected = selected.merge(
                stats[columns],
                on=[
                    "mon_thi",
                    "province_code"
                ],
                how="left",
                suffixes=("", "_stats")
            )

    columns_to_show = [
        "mon_thi",
        "province_name",
        "so_bai",
        "diem_tb",
        "js_divergence",
        "total_variation",
        "p_value_fdr"
    ]

    columns_to_show = [
        c for c in columns_to_show
        if c in selected.columns
    ]

    print(
        selected[columns_to_show]
        .head(20)
        .to_string(index=False)
    )

    selected.to_csv(
        OUTPUT / "anomalies_to_verify.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nTong so cap tinh-mon can xem:",
        len(selected)
    )


# ===================================================
# 7. MAIN
# ===================================================

if __name__ == "__main__":

    if not PARQUET.exists():
        raise FileNotFoundError(PARQUET)

    audit_columns()
    audit_anomalies()

    print("\nHOAN THANH KIEM TRA!")
