
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ==========================================================
# 1. CAU HINH
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "data" / "analysis" / "all_subjects"
DETAIL = ROOT / "data" / "analysis" / "detailed_scores"
KMEANS = ROOT / "data" / "analysis" / "kmeans_final"

OUTPUT = ROOT / "data" / "analysis" / "dashboard_tests"

SUBJECTS = {
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

EXPECTED_TOTALS = {
    "toan": 1183930,
    "ngu_van": 1188653,
    "ngoai_ngu": 342020,
    "vat_li": 382778,
    "hoa_hoc": 248576,
    "sinh_hoc": 68436,
    "lich_su": 561101,
    "dia_li": 440940,
    "gdkt_pl": 280675,
    "tin_hoc": 18141,
    "cong_nghe_nn": 30911,
    "cong_nghe_cn": 7162
}

FILES = {
    "national": BASE / "all_national_stats.csv",
    "province": BASE / "all_province_stats.csv",
    "national_dist": BASE / "all_national_distribution.csv",
    "province_dist": BASE / "all_province_distribution.csv",
    "anomalies": BASE / "all_anomaly_evaluation.csv",
    "bins_national": DETAIL / "all_binned_national.csv",
    "bins_province": DETAIL / "all_binned_province.csv",
    "exact_national": DETAIL / "all_detailed_national.csv",
    "exact_province": DETAIL / "all_detailed_province.csv",
    "spikes": DETAIL / "all_detailed_spikes.csv",
    "audit": DETAIL / "all_detailed_audit.csv",
    "km_results": KMEANS / "all_kmeans_results.csv",
    "km_eval": KMEANS / "all_kmeans_evaluation.csv",
    "km_selection": KMEANS / "kmeans_model_selection.csv"
}

RESULTS = []
DATA = {}


# ==========================================================
# 2. HAM KIEM THU
# ==========================================================

def check(group, name, condition, detail=""):

    passed = bool(condition)

    RESULTS.append({
        "nhom": group,
        "kiem_thu": name,
        "ket_qua": "PASS" if passed else "FAIL",
        "chi_tiet": str(detail)
    })

    symbol = "PASS" if passed else "FAIL"

    print(f"[{symbol}] {group} | {name}")

    if not passed and detail:
        print(f"       {detail}")


def numeric(df, column):

    if column not in df.columns:
        return pd.Series(dtype=float)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def subject_rows(df, subject):

    return df[df["mon_thi"] == subject]


def total_subject(df, subject):

    subset = subject_rows(df, subject)

    return int(
        numeric(subset, "so_bai").sum()
    )


def read_all_files():

    print("\n" + "=" * 65)
    print("KIEM TRA FILE CSV")
    print("=" * 65)

    failed_files = []

    for key, path in FILES.items():

        if not path.is_file():

            check(
                "FILE",
                path.name,
                False,
                "Khong tim thay file"
            )

            failed_files.append(path.name)
            continue

        try:

            df = pd.read_csv(
                path,
                encoding="utf-8-sig",
                dtype={
                    "province_code": "string"
                }
            )

            if "province_code" in df.columns:
                df["province_code"] = (
                    df["province_code"].str.zfill(2)
                )

            DATA[key] = df

            passed = not df.empty

            check(
                "FILE",
                path.name,
                passed,
                f"{len(df):,} dong"
            )

            if not passed:
                failed_files.append(path.name)

        except Exception as exc:

            check(
                "FILE",
                path.name,
                False,
                repr(exc)
            )

            failed_files.append(path.name)

    return failed_files


# ==========================================================
# 3. KIEM THU TONG QUAN
# ==========================================================

def test_overview():

    print("\n" + "=" * 65)
    print("KIEM TRA TONG QUAN")
    print("=" * 65)

    national = DATA["national"]
    province = DATA["province"]

    check(
        "TONG QUAN",
        "Du 12 mon thi",
        set(national["mon_thi"]) == set(SUBJECTS)
    )

    check(
        "TONG QUAN",
        "Du 34 dia phuong",
        province["province_code"].nunique() == 34
    )

    check(
        "TONG QUAN",
        "Khong trung thong ke mon",
        not national["mon_thi"].duplicated().any()
    )

    for subject, expected in EXPECTED_TOTALS.items():

        actual = total_subject(
            national,
            subject
        )

        check(
            "SO BAI",
            SUBJECTS[subject],
            actual == expected,
            f"Thuc te={actual:,}; "
            f"Ky vong={expected:,}"
        )


# ==========================================================
# 4. KIEM THU PHO DIEM 10 KHOANG
# ==========================================================

def test_10_bins():

    print("\n" + "=" * 65)
    print("KIEM TRA PHO DIEM 10 KHOANG")
    print("=" * 65)

    nat = DATA["national_dist"]
    prov = DATA["province_dist"]
    stats = DATA["province"]

    for subject in SUBJECTS:

        expected = EXPECTED_TOTALS[subject]

        a = subject_rows(nat, subject)
        b = subject_rows(prov, subject)
        c = subject_rows(stats, subject)

        check(
            "10 KHOANG",
            f"{subject}: tong toan quoc",
            int(numeric(a, "so_bai").sum()) == expected
        )

        check(
            "10 KHOANG",
            f"{subject}: tong dia phuong",
            int(numeric(b, "so_bai").sum()) == expected
        )

        # SUA LOI:
        # Khong bat buoc phai co du ca 10 khoang.
        # Mot khoang khong co bai thi co the vang mat.
        indices = numeric(a, "khoang_diem")

        check(
            "10 KHOANG",
            f"{subject}: chi so hop le 0 den 9",
            (
                indices.notna().all()
                and indices.between(0, 9).all()
                and (indices % 1 == 0).all()
                and not indices.duplicated().any()
            )
        )

        # Doi chieu tong bai thi theo tung tinh
        from_dist = b.groupby(
            "province_code"
        )["so_bai"].sum()

        from_stats = c.set_index(
            "province_code"
        )["so_bai"]

        aligned = from_dist.reindex(
            from_stats.index,
            fill_value=0
        )

        check(
            "10 KHOANG",
            f"{subject}: doi chieu theo tinh",
            (
                aligned.astype(int).equals(
                    from_stats.astype(int)
                )
                and set(from_dist.index)
                == set(from_stats.index)
            )
        )


# ==========================================================
# 5. KIEM THU PHO DIEM 40 KHOANG
# ==========================================================

def test_40_bins():

    print("\n" + "=" * 65)
    print("KIEM TRA PHO DIEM 40 KHOANG")
    print("=" * 65)

    nat = DATA["bins_national"]
    prov = DATA["bins_province"]
    audit = DATA["audit"]

    for subject in SUBJECTS:

        expected = EXPECTED_TOTALS[subject]

        a = subject_rows(nat, subject)
        b = subject_rows(prov, subject)
        c = subject_rows(audit, subject)

        national_total = int(
            numeric(a, "so_bai").sum()
        )

        provincial_total = int(
            numeric(b, "so_bai").sum()
        )

        check(
            "40 KHOANG",
            f"{subject}: toan quoc",
            national_total == expected
        )

        check(
            "40 KHOANG",
            f"{subject}: dia phuong",
            provincial_total == expected
        )

        indices = numeric(
            a, "bin_index"
        )

        check(
            "40 KHOANG",
            f"{subject}: chi so hop le",
            (
                indices.notna().all()
                and indices.between(0, 39).all()
                and (indices % 1 == 0).all()
                and not indices.duplicated().any()
            )
        )

        if not c.empty:

            audit_total = int(
                c.iloc[0]["tong_diem_hop_le"]
            )

            check(
                "AUDIT",
                f"{subject}: tong hop le",
                audit_total == expected
            )

            grid = int(
                c.iloc[0]["so_diem_tren_luoi"]
            )

            outside = int(
                c.iloc[0]["so_diem_ngoai_luoi"]
            )

            check(
                "AUDIT",
                f"{subject}: tren + ngoai luoi",
                grid + outside == expected
            )


# ==========================================================
# 6. KIEM THU JSD / TVD / FDR
# ==========================================================

def test_anomalies():

    print("\n" + "=" * 65)
    print("KIEM TRA JSD TVD FDR")
    print("=" * 65)

    df = DATA["anomalies"]

    # JSD va TVD phai day du va trong [0, 1].
    for column in [
        "js_divergence",
        "total_variation"
    ]:

        values = numeric(df, column)

        check(
            "ANOMALY",
            f"{column}: day du",
            (
                len(values) == len(df)
                and values.notna().all()
                and np.isfinite(values).all()
            )
        )

        check(
            "ANOMALY",
            f"{column}: trong [0,1]",
            values.between(0, 1).all()
        )

    # SUA LOI FDR:
    # Chi yeu cau FDR voi dia phuong co >= 500 bai.
    sample_sizes = numeric(df, "so_bai")
    fdr = numeric(df, "p_value_fdr")

    eligible = sample_sizes >= 500

    check(
        "ANOMALY",
        "p_value_fdr: day du voi mau >= 500",
        (
            fdr[eligible].notna().all()
            and np.isfinite(
                fdr[eligible]
            ).all()
        )
    )

    # Cac gia tri FDR hien co phai hop le.
    available_fdr = fdr.dropna()

    check(
        "ANOMALY",
        "p_value_fdr: trong [0,1]",
        (
            np.isfinite(
                available_fdr
            ).all()
            and available_fdr.between(
                0, 1
            ).all()
        )
    )

    check(
        "ANOMALY",
        "Khong trung mon va tinh",
        not df.duplicated(
            ["mon_thi", "province_code"]
        ).any()
    )

    filtered = df[
        (df["so_bai"] >= 500)
        & (df["js_divergence"] >= 0.02)
        & (df["total_variation"] >= 0.10)
        & (df["p_value_fdr"] < 0.05)
    ]

    print(
        "\nSo truong hop dap ung bo nguong:",
        len(filtered)
    )

    for subject in SUBJECTS:

        count = len(
            subject_rows(filtered, subject)
        )

        print(
            f"  {SUBJECTS[subject]}: {count}"
        )


# ==========================================================
# 7. KIEM THU K-MEANS CUOI CUNG
# ==========================================================

def test_kmeans():

    print("\n" + "=" * 65)
    print("KIEM TRA K-MEANS")
    print("=" * 65)

    results = DATA["km_results"]
    selection = DATA["km_selection"]
    evaluation = DATA["km_eval"]

    modeled = (
        set(SUBJECTS)
        - {"cong_nghe_cn"}
    )

    check(
        "KMEANS",
        "Du 11 mon co mo hinh",
        set(selection["mon_thi"]) == modeled
    )

    check(
        "KMEANS",
        "Cong nghe CN khong co mo hinh",
        (
            "cong_nghe_cn"
            not in set(results["mon_thi"])
            and "cong_nghe_cn"
            not in set(selection["mon_thi"])
        )
    )

    check(
        "KMEANS",
        "Khong trung mon va tinh",
        not results.duplicated(
            ["mon_thi", "province_code"]
        ).any()
    )

    for subject in modeled:

        selected = subject_rows(
            selection, subject
        )

        sub = subject_rows(
            results, subject
        )

        ev = subject_rows(
            evaluation, subject
        )

        if (
            selected.empty
            or sub.empty
            or ev.empty
        ):

            check(
                "KMEANS",
                subject,
                False,
                "Thieu ket qua"
            )

            continue

        row = selected.iloc[0]

        k = int(row["k"])
        seed = int(row["selected_seed"])
        ari = float(row["ari_mean"])
        sil = float(row["silhouette"])

        labels = set(
            pd.to_numeric(
                sub["cluster"]
            ).astype(int)
        )

        check(
            "KMEANS",
            f"{subject}: so cum",
            (
                len(labels) == k
                and labels == set(range(k))
            )
        )

        check(
            "KMEANS",
            f"{subject}: seed va K",
            (
                (sub["best_k"] == k).all()
                and (
                    sub["selected_seed"]
                    == seed
                ).all()
            )
        )

        check(
            "KMEANS",
            f"{subject}: ARI Silhouette",
            (
                0 <= ari <= 1
                and -1 <= sil <= 1
            )
        )

        sizes = sub.groupby(
            "cluster"
        ).size()

        check(
            "KMEANS",
            f"{subject}: it nhat 3 tinh moi cum",
            (sizes >= 3).all()
        )

    # Truong hop da xac nhan
    math = subject_rows(
        selection, "toan"
    )

    literature = subject_rows(
        selection, "ngu_van"
    )

    if not math.empty:

        check(
            "KMEANS",
            "Toan: K=3 Seed=22",
            (
                int(math.iloc[0]["k"]) == 3
                and int(
                    math.iloc[0]["selected_seed"]
                ) == 22
            )
        )

    if not literature.empty:

        check(
            "KMEANS",
            "Ngu van: K=2 Seed=33",
            (
                int(literature.iloc[0]["k"]) == 2
                and int(
                    literature.iloc[0]["selected_seed"]
                ) == 33
            )
        )


# ==========================================================
# 8. KIEM THU SPIKE RATIO
# ==========================================================

def test_spikes():

    print("\n" + "=" * 65)
    print("KIEM TRA SPIKE RATIO")
    print("=" * 65)

    spikes = DATA["spikes"]

    for column in [
        "so_bai_tai_diem",
        "so_bai_ben_trai",
        "so_bai_ben_phai",
        "spike_ratio"
    ]:

        values = numeric(
            spikes, column
        )

        check(
            "SPIKE",
            f"{column}: khong am va huu han",
            (
                values.notna().all()
                and np.isfinite(values).all()
                and (values >= 0).all()
            )
        )

    neighbors = (
        numeric(
            spikes,
            "so_bai_ben_trai"
        )
        +
        numeric(
            spikes,
            "so_bai_ben_phai"
        )
    ) / 2

    ratios = numeric(
        spikes,
        "spike_ratio"
    )

    calculated = (
        numeric(
            spikes,
            "so_bai_tai_diem"
        )
        / neighbors.replace(
            0, np.nan
        )
    )

    valid = neighbors > 0

    check(
        "SPIKE",
        "Ty le tinh dung cong thuc",
        (
            valid.all()
            and np.allclose(
                calculated.to_numpy(),
                ratios.to_numpy(),
                rtol=1e-8,
                atol=1e-8
            )
        )
    )

    exact = DATA["exact_national"]
    audit = DATA["audit"]

    for subject in SUBJECTS:

        actual = total_subject(
            exact, subject
        )

        row = subject_rows(
            audit, subject
        )

        expected = int(
            row.iloc[0][
                "so_diem_tren_luoi"
            ]
        )

        check(
            "SPIKE",
            f"{subject}: tong diem tren luoi",
            actual == expected
        )


# ==========================================================
# 9. LUU BAO CAO KIEM THU
# ==========================================================

def save_report():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(
        RESULTS
    )

    path = (
        OUTPUT
        / "dashboard_test_results.csv"
    )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    passed = int(
        (df["ket_qua"] == "PASS").sum()
    )

    failed = int(
        (df["ket_qua"] == "FAIL").sum()
    )

    print("\n" + "=" * 65)
    print("KET QUA KIEM THU DASHBOARD")
    print("=" * 65)

    print(
        "Tong bai kiem thu:",
        len(df)
    )

    print(
        "PASS:",
        passed
    )

    print(
        "FAIL:",
        failed
    )

    print("\nDa luu bao cao:")
    print(path)

    if failed:

        print("\nCAC BAI FAIL:")

        print(
            df[
                df["ket_qua"] == "FAIL"
            ].to_string(index=False)
        )

    else:

        print(
            "\nTAT CA BAI KIEM THU DEU PASS!"
        )

    return failed


# ==========================================================
# 10. MAIN
# ==========================================================

def main():

    print("=" * 65)
    print("THPT ANALYTICS 2026 - DATA TESTS")
    print("=" * 65)

    missing = read_all_files()

    if missing:

        print(
            "\nDung kiem tra du lieu vi "
            "co file thieu hoac rong."
        )

        save_report()
        return 1

    test_overview()
    test_10_bins()
    test_40_bins()
    test_anomalies()
    test_kmeans()
    test_spikes()

    failed = save_report()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
