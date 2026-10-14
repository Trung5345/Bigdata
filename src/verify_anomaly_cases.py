
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib

# Luu bieu do truc tiep, khong mo cua so
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =====================================================
# 1. CAU HINH
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT / "data" / "analysis" / "all_subjects"
)

OUTPUT_DIR = (
    ROOT / "data" / "analysis" / "anomaly_cases"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANOMALY_FILE = (
    DATA_DIR / "all_anomaly_evaluation.csv"
)

DISTRIBUTION_FILE = (
    DATA_DIR / "all_province_distribution.csv"
)

STATS_FILE = (
    DATA_DIR / "all_province_stats.csv"
)

TOP_N = 5
MIN_SAMPLES = 500
MIN_JSD = 0.02
MIN_TVD = 0.10
MAX_FDR = 0.05

BIN_LABELS = [
    "0–<1",
    "1–<2",
    "2–<3",
    "3–<4",
    "4–<5",
    "5–<6",
    "6–<7",
    "7–<8",
    "8–<9",
    "9–10"
]

SUBJECT_NAMES = {
    "toan": "Toán",
    "ngu_van": "Ngữ văn",
    "ngoai_ngu": "Ngoại ngữ",
    "vat_li": "Vật lý",
    "hoa_hoc": "Hóa học",
    "sinh_hoc": "Sinh học",
    "lich_su": "Lịch sử",
    "dia_li": "Địa lý",
    "gdkt_pl": "Giáo dục KT&PL",
    "tin_hoc": "Tin học",
    "cong_nghe_nn": "Công nghệ nông nghiệp",
    "cong_nghe_cn": "Công nghệ công nghiệp"
}


# =====================================================
# 2. DOC DU LIEU
# =====================================================

def load_data():

    required = [
        ANOMALY_FILE,
        DISTRIBUTION_FILE,
        STATS_FILE
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Khong tim thay file: {path}"
            )

    anomalies = pd.read_csv(
        ANOMALY_FILE,
        dtype={"province_code": str}
    )

    distributions = pd.read_csv(
        DISTRIBUTION_FILE,
        dtype={"province_code": str}
    )

    stats = pd.read_csv(
        STATS_FILE,
        dtype={"province_code": str}
    )

    for df in [
        anomalies,
        distributions,
        stats
    ]:
        df["province_code"] = (
            df["province_code"]
            .str.zfill(2)
        )

    return anomalies, distributions, stats


# =====================================================
# 3. CHON CAC TRUONG HOP NOI BAT
# =====================================================

def select_cases(anomalies):

    selected = anomalies[
        (anomalies["so_bai"] >= MIN_SAMPLES)
        & (
            anomalies["js_divergence"]
            >= MIN_JSD
        )
        & (
            anomalies["total_variation"]
            >= MIN_TVD
        )
        & (
            anomalies["p_value_fdr"]
            < MAX_FDR
        )
    ].copy()

    selected = selected.sort_values(
        "js_divergence",
        ascending=False
    )

    selected = selected.head(TOP_N)

    return selected


# =====================================================
# 4. LAY PHO DIEM CUA DIA PHUONG
#    VA CAC DIA PHUONG CON LAI
# =====================================================

def get_distributions(
    distributions,
    subject,
    province_code
):

    subject_df = distributions[
        distributions["mon_thi"] == subject
    ].copy()

    province_df = subject_df[
        subject_df["province_code"]
        == province_code
    ]

    rest_df = subject_df[
        subject_df["province_code"]
        != province_code
    ]

    province_counts = (
        province_df.groupby("khoang_diem")[
            "so_bai"
        ]
        .sum()
        .reindex(range(10), fill_value=0)
    )

    rest_counts = (
        rest_df.groupby("khoang_diem")[
            "so_bai"
        ]
        .sum()
        .reindex(range(10), fill_value=0)
    )

    province_total = province_counts.sum()
    rest_total = rest_counts.sum()

    if province_total <= 0:
        raise ValueError(
            "Dia phuong khong co bai thi hop le."
        )

    if rest_total <= 0:
        raise ValueError(
            "Khong co du lieu nhom doi chieu."
        )

    province_pct = (
        province_counts
        / province_total * 100
    )

    rest_pct = (
        rest_counts
        / rest_total * 100
    )

    return (
        province_counts,
        rest_counts,
        province_pct,
        rest_pct
    )


# =====================================================
# 5. PHAN TICH CHENH LECH
# =====================================================

def analyze_case(
    distributions,
    subject,
    province_code,
    province_name
):

    (
        province_counts,
        rest_counts,
        province_pct,
        rest_pct
    ) = get_distributions(
        distributions,
        subject,
        province_code
    )

    difference = (
        province_pct - rest_pct
    )

    abs_difference = difference.abs()

    result = pd.DataFrame({
        "mon_thi": subject,
        "province_code": province_code,
        "province_name": province_name,
        "khoang_diem": list(range(10)),
        "nhan_khoang_diem": BIN_LABELS,
        "so_bai_dia_phuong": (
            province_counts.to_numpy()
        ),
        "so_bai_cac_tinh_con_lai": (
            rest_counts.to_numpy()
        ),
        "ty_le_dia_phuong_pct": (
            province_pct.to_numpy()
        ),
        "ty_le_cac_tinh_con_lai_pct": (
            rest_pct.to_numpy()
        ),
        "chenh_lech_diem_phan_tram": (
            difference.to_numpy()
        ),
        "chenh_lech_tuyet_doi": (
            abs_difference.to_numpy()
        )
    })

    max_row = result.loc[
        result["chenh_lech_tuyet_doi"]
        .idxmax()
    ]

    # Kiem tra TVD tu du lieu phan bo
    tvd_recalculated = (
        abs_difference.sum() / 200
    )

    summary = {
        "mon_thi": subject,
        "ten_mon": SUBJECT_NAMES.get(
            subject,
            subject
        ),
        "province_code": province_code,
        "province_name": province_name,
        "so_bai_dia_phuong": int(
            province_counts.sum()
        ),
        "so_bai_nhom_con_lai": int(
            rest_counts.sum()
        ),
        "khoang_khac_biet_nhat": (
            max_row["nhan_khoang_diem"]
        ),
        "ty_le_dia_phuong_pct": float(
            max_row["ty_le_dia_phuong_pct"]
        ),
        "ty_le_nhom_con_lai_pct": float(
            max_row["ty_le_cac_tinh_con_lai_pct"]
        ),
        "chenh_lech_lon_nhat_pp": float(
            max_row["chenh_lech_diem_phan_tram"]
        ),
        "tvd_tinh_lai": float(
            tvd_recalculated
        )
    }

    return result, summary


# =====================================================
# 6. VE BIEU DO SO SANH PHO DIEM
# =====================================================

def plot_distribution(
    detail,
    subject,
    province_name,
    output_path
):

    x = np.arange(10)
    width = 0.38

    fig, ax = plt.subplots(
        figsize=(12, 6.5)
    )

    ax.bar(
        x - width / 2,
        detail["ty_le_dia_phuong_pct"],
        width,
        label=province_name,
        color="#2563eb"
    )

    ax.bar(
        x + width / 2,
        detail["ty_le_cac_tinh_con_lai_pct"],
        width,
        label="Cac dia phuong con lai",
        color="#94a3b8"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)

    ax.set_xlabel("Khoang diem")
    ax.set_ylabel("Ty le bai thi (%)")

    ax.set_title(
        f"So sanh pho diem - "
        f"{province_name} - "
        f"{SUBJECT_NAMES.get(subject, subject)}"
    )

    ax.legend()

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# =====================================================
# 7. VE BIEU DO CHENH LECH
# =====================================================

def plot_difference(
    detail,
    subject,
    province_name,
    output_path
):

    x = np.arange(10)

    difference = detail[
        "chenh_lech_diem_phan_tram"
    ].to_numpy()

    colors = [
        "#2563eb" if value >= 0
        else "#ef4444"
        for value in difference
    ]

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.bar(
        x,
        difference,
        color=colors
    )

    ax.axhline(
        0,
        color="#334155",
        linewidth=1
    )

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)

    ax.set_xlabel("Khoang diem")
    ax.set_ylabel(
        "Chenh lech (diem phan tram)"
    )

    ax.set_title(
        f"Chenh lech pho diem - "
        f"{province_name} - "
        f"{SUBJECT_NAMES.get(subject, subject)}"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# =====================================================
# 8. MAIN
# =====================================================

def main():

    print("=" * 70)
    print("PHAN TICH CHI TIET CAC TRUONG HOP KHAC BIET")
    print("=" * 70)

    anomalies, distributions, stats = (
        load_data()
    )

    cases = select_cases(anomalies)

    if cases.empty:
        print(
            "Khong co truong hop nao "
            "thoa nguong."
        )
        return

    all_details = []
    all_summaries = []

    for index, case in enumerate(
        cases.itertuples(index=False),
        start=1
    ):

        subject = str(case.mon_thi)
        province_code = str(
            case.province_code
        )
        province_name = str(
            case.province_name
        )

        print("\n" + "-" * 65)

        print(
            f"TRUONG HOP {index}: "
            f"{province_name} - "
            f"{SUBJECT_NAMES.get(subject, subject)}"
        )

        print(
            "JSD:",
            round(
                float(case.js_divergence),
                6
            )
        )

        detail, summary = analyze_case(
            distributions,
            subject,
            province_code,
            province_name
        )

        summary["js_divergence"] = float(
            case.js_divergence
        )

        summary["tvd_goc"] = float(
            case.total_variation
        )

        summary["p_value_fdr"] = float(
            case.p_value_fdr
        )

        # Lay diem trung binh tu bang thong ke
        matched_stats = stats[
            (stats["mon_thi"] == subject)
            & (
                stats["province_code"]
                == province_code
            )
        ]

        summary["diem_tb"] = (
            float(
                matched_stats["diem_tb"]
                .iloc[0]
            )
            if not matched_stats.empty
            else np.nan
        )

        # Doi chieu TVD tinh lai voi file goc
        tvd_diff = abs(
            summary["tvd_tinh_lai"]
            - summary["tvd_goc"]
        )

        summary["sai_lech_tvd"] = tvd_diff

        if tvd_diff > 1e-5:
            print(
                "CANH BAO: TVD tinh lai "
                "khac TVD trong file goc:",
                round(tvd_diff, 6)
            )

        all_details.append(detail)
        all_summaries.append(summary)

        file_base = (
            f"{index:02d}_{subject}_"
            f"{province_code}"
        )

        # Luu bang chi tiet rieng
        detail.to_csv(
            OUTPUT_DIR / f"{file_base}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        # Bieu do so sanh
        plot_distribution(
            detail,
            subject,
            province_name,
            OUTPUT_DIR
            / f"{file_base}_distribution.png"
        )

        # Bieu do chenh lech
        plot_difference(
            detail,
            subject,
            province_name,
            OUTPUT_DIR
            / f"{file_base}_difference.png"
        )

        print(
            "So bai dia phuong:",
            summary["so_bai_dia_phuong"]
        )

        print(
            "Khoang khac biet nhat:",
            summary["khoang_khac_biet_nhat"]
        )

        print(
            "Chenh lech lon nhat:",
            round(
                summary[
                    "chenh_lech_lon_nhat_pp"
                ],
                3
            ),
            "diem phan tram"
        )

        print(
            "TVD tinh lai:",
            round(
                summary["tvd_tinh_lai"],
                6
            )
        )

    # ================================================
    # 9. LUU BAO CAO TONG HOP
    # ================================================

    summary_df = pd.DataFrame(
        all_summaries
    )

    detail_df = pd.concat(
        all_details,
        ignore_index=True
    )

    summary_file = (
        OUTPUT_DIR / "anomaly_cases_summary.csv"
    )

    detail_file = (
        OUTPUT_DIR / "anomaly_cases_details.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    detail_df.to_csv(
        detail_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("TONG HOP PHAN TICH")
    print("=" * 70)

    display_cols = [
        "province_name",
        "ten_mon",
        "diem_tb",
        "js_divergence",
        "tvd_tinh_lai",
        "khoang_khac_biet_nhat",
        "chenh_lech_lon_nhat_pp"
    ]

    print(
        summary_df[display_cols]
        .round(4)
        .to_string(index=False)
    )

    print("\nDa luu ket qua tai:")
    print(OUTPUT_DIR)

    print("\nHOAN THANH!")


if __name__ == "__main__":
    main()
