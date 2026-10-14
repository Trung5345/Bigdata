
from pathlib import Path
import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. CAU HINH
# ============================================================

st.set_page_config(
    page_title="THPT Analytics 2026",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "data" / "analysis" / "all_subjects"
KMEANS_DIR = ROOT / "data" / "analysis" / "kmeans_final"
DETAIL_DIR = ROOT / "data" / "analysis" / "detailed_scores"
CASE_DIR = ROOT / "data" / "analysis" / "anomaly_cases"

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

BINS = [
    f"{i}–<{i + 1}" for i in range(9)
] + ["9–10"]

COLORS = [
    "#2563eb", "#0d9488", "#e5a339",
    "#8b5cf6", "#df6676", "#329acb"
]


# ============================================================
# 2. CSS
# ============================================================

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #f4f7fb;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e4eaf2;
    min-width: 295px !important;
    max-width: 295px !important;
}

[data-testid="stSidebarUserContent"] {
    padding-top: 25px;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 4px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 12px 13px;
    border-radius: 9px;
    min-height: 49px;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #f2f6fc;
}

[data-testid="stSidebar"]
[role="radiogroup"] label:has(input:checked) {
    background: #e9f1ff;
}

[data-testid="stSidebar"] [role="radiogroup"] label p {
    font-size: 15px !important;
    color: #52647c !important;
}

[data-testid="stSidebar"]
[role="radiogroup"] label:has(input:checked) p {
    color: #1e57cd !important;
    font-weight: 750 !important;
}

[data-testid="stSidebar"]
[role="radiogroup"] [role="radio"] {
    display: none !important;
}

[data-testid="stSidebar"]
[data-testid="stWidgetLabel"] p {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #6c7d94 !important;
}

[data-testid="stSidebar"]
[data-baseweb="select"] > div {
    border-radius: 9px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5ebf3;
    border-radius: 13px;
    padding: 17px 18px;
}

[data-testid="stMetricValue"] {
    color: #255acb;
    font-weight: 800;
}

[data-testid="stPlotlyChart"] {
    background: white;
    border: 1px solid #e5ebf3;
    border-radius: 13px;
    padding: 9px;
}

h1, h2, h3 {
    color: #253751;
}

.brand {
    display: flex;
    gap: 12px;
    align-items: center;
    padding-bottom: 20px;
    border-bottom: 1px solid #edf1f5;
    margin-bottom: 20px;
}

.brand-icon {
    width: 45px;
    height: 45px;
    border-radius: 10px;
    background: #2563eb;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 800;
    font-size: 20px;
}

.brand-title {
    color: #1e4fae;
    font-size: 17px;
    font-weight: 800;
}

.brand-subtitle {
    color: #8492a7;
    font-size: 11px;
    line-height: 1.7;
}

.topbar {
    background: white;
    border: 1px solid #e5ebf3;
    border-radius: 13px;
    padding: 17px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
}

.topbar-title {
    color: #263852;
    font-size: 14px;
    font-weight: 750;
}

.topbar-sub {
    color: #8492a7;
    font-size: 12px;
}

.avatar {
    background: #eaf1ff;
    color: #245cce;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    font-weight: 800;
}

.page-title {
    font-size: 27px;
    font-weight: 800;
    color: #253751;
    margin-bottom: 6px;
}

.page-desc {
    font-size: 13px;
    color: #718098;
    margin-bottom: 23px;
}

.small-note {
    padding: 13px 16px;
    background: #edf4ff;
    border-left: 3px solid #3478ef;
    border-radius: 8px;
    color: #49678e;
    font-size: 13px;
    margin: 15px 0 22px;
    line-height: 1.8;
}

.footer {
    color: #8795a8;
    font-size: 12px;
    padding-top: 18px;
    margin-top: 30px;
    border-top: 1px solid #e4eaf2;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. DOC FILE
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv(path):

    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"province_code": str}
    )

    if "province_code" in df.columns:
        df["province_code"] = (
            df["province_code"].str.zfill(2)
        )

    return df


national = read_csv(
    BASE / "all_national_stats.csv"
)

province = read_csv(
    BASE / "all_province_stats.csv"
)

national_dist = read_csv(
    BASE / "all_national_distribution.csv"
)

province_dist = read_csv(
    BASE / "all_province_distribution.csv"
)

anomalies = read_csv(
    BASE / "all_anomaly_evaluation.csv"
)

clusters = read_csv(
    KMEANS_DIR / "all_kmeans_results.csv"
)

cluster_eval = read_csv(
    KMEANS_DIR / "all_kmeans_evaluation.csv"
)

cluster_selection = read_csv(
    KMEANS_DIR / "kmeans_model_selection.csv"
)

binned_national = read_csv(
    DETAIL_DIR / "all_binned_national.csv"
)

binned_province = read_csv(
    DETAIL_DIR / "all_binned_province.csv"
)

exact_national = read_csv(
    DETAIL_DIR / "all_detailed_national.csv"
)

exact_province = read_csv(
    DETAIL_DIR / "all_detailed_province.csv"
)

detailed_spikes = read_csv(
    DETAIL_DIR / "all_detailed_spikes.csv"
)

detailed_audit = read_csv(
    DETAIL_DIR / "all_detailed_audit.csv"
)

case_summary = read_csv(
    CASE_DIR / "anomaly_cases_summary.csv"
)


# ============================================================
# 4. KIEM TRA FILE
# ============================================================

required = {
    "all_national_stats.csv": national,
    "all_province_stats.csv": province,
    "all_national_distribution.csv": national_dist,
    "all_province_distribution.csv": province_dist,
    "all_anomaly_evaluation.csv": anomalies
}

missing = [
    filename
    for filename, data in required.items()
    if data.empty
]

if missing:
    st.error(
        "Thiếu dữ liệu: " + ", ".join(missing)
    )
    st.code(str(BASE))
    st.stop()


# ============================================================
# 5. HAM HO TRO
# ============================================================

def subject_name(code):
    return SUBJECTS.get(code, code)


def page_heading(title, description):

    st.markdown(
        f"""
        <div class="page-title">
            {html.escape(title)}
        </div>
        <div class="page-desc">
            {html.escape(description)}
        </div>
        """,
        unsafe_allow_html=True
    )


def note(text):

    st.markdown(
        f"""
        <div class="small-note">
            {html.escape(text)}
        </div>
        """,
        unsafe_allow_html=True
    )


def style_chart(fig, height=420, legend=False):

    # Tieu de dat ben ngoai bieu do
    # de khong de len legend.

    fig.update_layout(
        title=None,
        template="plotly_white",
        height=height,
        margin=dict(
            l=35,
            r=25,
            t=85 if legend else 30,
            b=50
        ),
        font=dict(
            family="Arial",
            size=12,
            color="#52637a"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=legend,
        legend=dict(
            orientation="h",
            x=0,
            y=1.15,
            xanchor="left",
            yanchor="bottom"
        )
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        automargin=True
    )

    fig.update_yaxes(
        gridcolor="#edf1f6",
        zeroline=False,
        automargin=True
    )

    return fig


def show_chart(fig, height=420, legend=False):

    st.plotly_chart(
        style_chart(fig, height, legend),
        use_container_width=True
    )


def download_csv(df, filename):

    if df.empty:
        return

    st.download_button(
        "Tải kết quả CSV",
        data=df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv"
    )


def labeled_distribution(df):

    df = df.copy()

    df["khoang_diem"] = pd.to_numeric(
        df["khoang_diem"]
    ).astype(int)

    df["Khoảng điểm"] = df[
        "khoang_diem"
    ].map(
        dict(enumerate(BINS))
    )

    return df


def compare_rest(subject, province_code):

    data = province_dist[
        province_dist["mon_thi"] == subject
    ].copy()

    target = (
        data[
            data["province_code"]
            == province_code
        ]
        .groupby("khoang_diem")["so_bai"]
        .sum()
        .reindex(range(10), fill_value=0)
    )

    rest = (
        data[
            data["province_code"]
            != province_code
        ]
        .groupby("khoang_diem")["so_bai"]
        .sum()
        .reindex(range(10), fill_value=0)
    )

    if target.sum() == 0 or rest.sum() == 0:
        return pd.DataFrame()

    return pd.DataFrame({
        "Khoảng điểm": BINS,
        "Địa phương": (
            100 * target.to_numpy() / target.sum()
        ),
        "Các địa phương còn lại": (
            100 * rest.to_numpy() / rest.sum()
        )
    })


def bin_labels():

    result = {}

    for i in range(40):

        lower = i * 0.25
        upper = (i + 1) * 0.25

        if i == 39:
            result[i] = (
                f"{lower:.2f}–{upper:.2f}"
            )
        else:
            result[i] = (
                f"{lower:.2f}–<{upper:.2f}"
            )

    return result


BIN_LABELS = bin_labels()
ALL_BINS = pd.Index(range(40))


def get_bin_counts(data):

    return (
        data.groupby("bin_index")["so_bai"]
        .sum()
        .reindex(ALL_BINS, fill_value=0)
        .to_numpy(dtype=float)
    )


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="brand">
        <div class="brand-icon">TH</div>
        <div>
            <div class="brand-title">
                THPT ANALYTICS
            </div>
            <div class="brand-subtitle">
                HỆ THỐNG PHÂN TÍCH PHỔ ĐIỂM
                <br>
                NĂM 2026
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        "Tổng quan",
        "So sánh phổ điểm",
        "Phát hiện khác biệt",
        "Phân cụm K-means",
        "Phân tích chuyên sâu"
    ]

    page = st.radio(
        "ĐIỀU HƯỚNG",
        pages
    )

    st.divider()

    subjects = sorted(
        national["mon_thi"]
        .dropna()
        .unique(),
        key=subject_name
    )

    default_index = (
        subjects.index("ngu_van")
        if "ngu_van" in subjects
        else 0
    )

    subject = st.selectbox(
        "CHỌN MÔN THI",
        subjects,
        index=default_index,
        format_func=subject_name
    )

    st.caption(
        "ĐỒ ÁN NHẬP MÔN BIG DATA"
    )

    st.caption(
        "Apache Spark · Spark MLlib · Streamlit"
    )


# ============================================================
# 7. TOPBAR
# ============================================================

st.markdown("""
<div class="topbar">
    <div>
        <div class="topbar-title">
            Phân tích phổ điểm THPT 2026
        </div>
        <div class="topbar-sub">
            Bảng điều khiển nghiên cứu
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:12px">
        <div style="text-align:right">
            <div class="topbar-title">
                Nhóm Big Data
            </div>
            <div class="topbar-sub">
                Đồ án môn học
            </div>
        </div>
        <div class="avatar">BD</div>
    </div>
</div>
""", unsafe_allow_html=True)

subject_label = subject_name(subject)

ps = province[
    province["mon_thi"] == subject
].copy()

ns = national[
    national["mon_thi"] == subject
].copy()

nds = national_dist[
    national_dist["mon_thi"] == subject
].copy()

pds = province_dist[
    province_dist["mon_thi"] == subject
].copy()

ans = anomalies[
    anomalies["mon_thi"] == subject
].copy()


# ============================================================
# 8. TRANG TONG QUAN
# ============================================================

if page == "Tổng quan":

    page_heading(
        "Tổng quan dữ liệu",
        f"Thống kê kết quả thi THPT 2026 – "
        f"Môn {subject_label}"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Bản ghi thí sinh",
        "1.200.202"
    )

    b.metric(
        "Tỉnh/thành phố",
        province["province_code"].nunique()
    )

    c.metric(
        "Môn thi",
        national["mon_thi"].nunique()
    )

    d.metric(
        f"Bài thi {subject_label}",
        f"{int(nds['so_bai'].sum()):,}"
    )

    note(
        "Phổ điểm tổng quan được chia thành "
        "10 khoảng điểm từ 0 đến 10. "
        "Số bài thi khác nhau giữa các môn."
    )

    left, right = st.columns([1.2, 1])

    with left:

        st.subheader(
            f"Phổ điểm toàn quốc – {subject_label}"
        )

        chart = labeled_distribution(
            nds
        ).sort_values("khoang_diem")

        fig = px.bar(
            chart,
            x="Khoảng điểm",
            y="so_bai",
            labels={
                "so_bai": "Số bài thi"
            },
            color_discrete_sequence=[
                "#2563eb"
            ],
            category_orders={
                "Khoảng điểm": BINS
            }
        )

        show_chart(fig)

    with right:

        st.subheader(
            "10 địa phương có điểm trung bình cao"
        )

        top = (
            ps.nlargest(10, "diem_tb")
            .sort_values("diem_tb")
        )

        fig = px.bar(
            top,
            x="diem_tb",
            y="province_name",
            orientation="h",
            labels={
                "diem_tb": "Điểm trung bình",
                "province_name": ""
            },
            color_discrete_sequence=[
                "#329acb"
            ]
        )

        fig.update_xaxes(
            range=[0, 10]
        )

        show_chart(fig)

    st.subheader(
        "Thống kê môn thi"
    )

    st.dataframe(
        ns,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 9. TRANG SO SANH PHO DIEM
# ============================================================

elif page == "So sánh phổ điểm":

    page_heading(
        "So sánh phổ điểm",
        f"Đối chiếu phân bố điểm "
        f"giữa các địa phương – {subject_label}"
    )

    locations = sorted(
        pds["province_name"]
        .dropna()
        .unique()
        .tolist()
    )

    preferred = [
        x for x in [
            "Vĩnh Long",
            "Tây Ninh",
            "Sơn La"
        ]
        if x in locations
    ]

    default = (
        preferred
        if subject == "ngu_van"
        else locations[:3]
    )

    selected = st.multiselect(
        "Chọn tối đa 6 địa phương",
        locations,
        default=default,
        max_selections=6
    )

    show_national = st.checkbox(
        "Hiển thị phổ điểm toàn quốc",
        value=True
    )

    if not selected:

        st.info(
            "Hãy chọn ít nhất một địa phương."
        )

    else:

        selected_data = labeled_distribution(
            pds[
                pds["province_name"].isin(selected)
            ]
        )

        totals = selected_data.groupby(
            "province_name"
        )["so_bai"].transform("sum")

        selected_data["ty_le"] = (
            selected_data["so_bai"]
            / totals * 100
        )

        fig = px.line(
            selected_data,
            x="Khoảng điểm",
            y="ty_le",
            color="province_name",
            markers=True,
            category_orders={
                "Khoảng điểm": BINS
            },
            labels={
                "ty_le": "Tỷ lệ (%)",
                "province_name": "Địa phương"
            },
            color_discrete_sequence=COLORS
        )

        if show_national:

            nat_data = labeled_distribution(
                nds
            ).sort_values("khoang_diem")

            fig.add_trace(
                go.Scatter(
                    x=nat_data["Khoảng điểm"],
                    y=(
                        nat_data["so_bai"]
                        / nat_data["so_bai"].sum()
                        * 100
                    ),
                    name="Toàn quốc",
                    mode="lines+markers",
                    line=dict(
                        color="#27364d",
                        dash="dash",
                        width=3
                    )
                )
            )

        st.subheader(
            "So sánh tỷ lệ phân bố điểm"
        )

        show_chart(
            fig,
            height=490,
            legend=True
        )

        st.subheader(
            "Thống kê địa phương"
        )

        details = ps[
            ps["province_name"].isin(selected)
        ].sort_values(
            "diem_tb",
            ascending=False
        )

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True
        )

        download_csv(
            details,
            f"so_sanh_{subject}.csv"
        )


# ============================================================
# 10. TRANG PHAT HIEN KHAC BIET
# ============================================================

elif page == "Phát hiện khác biệt":

    page_heading(
        "Phát hiện phân bố khác biệt",
        f"Phân tích JSD, TVD và FDR – "
        f"Môn {subject_label}"
    )

    note(
        "Kết quả chỉ phản ánh sự khác biệt "
        "thống kê so với các địa phương còn lại. "
        "Không dùng để kết luận gian lận."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        min_n = st.number_input(
            "Số bài thi tối thiểu",
            min_value=100,
            max_value=100000,
            value=500,
            step=100
        )

    with c2:

        min_jsd = st.slider(
            "Ngưỡng JSD",
            0.0,
            0.3,
            0.02,
            0.005
        )

    with c3:

        min_tvd = st.slider(
            "Ngưỡng TVD",
            0.0,
            0.6,
            0.1,
            0.01
        )

    flagged = ans[
        (ans["so_bai"] >= min_n)
        & (
            ans["js_divergence"]
            >= min_jsd
        )
        & (
            ans["total_variation"]
            >= min_tvd
        )
        & (
            ans["p_value_fdr"] < 0.05
        )
    ].sort_values(
        "js_divergence",
        ascending=False
    )

    a, b, c = st.columns(3)

    a.metric(
        "Địa phương cần xem xét",
        len(flagged)
    )

    b.metric(
        "Địa phương đủ mẫu",
        int(
            (ans["so_bai"] >= min_n).sum()
        )
    )

    c.metric(
        "JSD lớn nhất",
        (
            f"{ans['js_divergence'].max():.4f}"
            if not ans.empty
            else "—"
        )
    )

    with st.expander(
        "Giải thích các chỉ số"
    ):

        st.markdown("""
        **JSD (Jensen-Shannon Divergence):**
        Đo mức khác biệt giữa hai phân bố xác suất.
        Giá trị càng lớn, hai phổ điểm càng khác nhau.

        **TVD (Total Variation Distance):**
        Đo tổng mức chênh lệch tỷ lệ giữa
        hai phân bố.

        **FDR:**
        Giá trị p đã hiệu chỉnh khi kiểm định
        đồng thời nhiều địa phương.

        Các chỉ số dùng để sàng lọc trường hợp
        cần kiểm tra sâu hơn, không xác định
        nguyên nhân của khác biệt.
        """)

    if flagged.empty:

        st.info(
            "Không có địa phương nào "
            "đáp ứng các ngưỡng hiện tại."
        )

    else:

        st.subheader(
            "Xếp hạng mức độ khác biệt"
        )

        ranking = (
            flagged.head(20)
            .sort_values("js_divergence")
        )

        fig = px.bar(
            ranking,
            x="js_divergence",
            y="province_name",
            orientation="h",
            labels={
                "js_divergence": "JSD",
                "province_name": ""
            },
            color_discrete_sequence=[
                COLORS[0]
            ]
        )

        show_chart(
            fig,
            max(380, len(ranking) * 35)
        )

        st.subheader(
            "Phân tích một địa phương"
        )

        chosen_code = st.selectbox(
            "Chọn địa phương",
            flagged["province_code"].tolist(),
            format_func=lambda code: (
                flagged.loc[
                    flagged["province_code"] == code,
                    "province_name"
                ].iloc[0]
            )
        )

        comparison = compare_rest(
            subject,
            chosen_code
        )

        if not comparison.empty:

            comparison["Chênh lệch"] = (
                comparison["Địa phương"]
                - comparison[
                    "Các địa phương còn lại"
                ]
            )

            fig = px.bar(
                comparison,
                x="Khoảng điểm",
                y="Chênh lệch",
                color_discrete_sequence=[
                    COLORS[0]
                ]
            )

            fig.add_hline(
                y=0,
                line_color="#64748b"
            )

            fig.update_yaxes(
                title="Chênh lệch (điểm phần trăm)"
            )

            show_chart(fig, 370)

        st.subheader(
            "Danh sách địa phương cần xem xét"
        )

        st.dataframe(
            flagged,
            use_container_width=True,
            hide_index=True
        )

        download_csv(
            flagged,
            f"anomaly_{subject}.csv"
        )


# ============================================================
# 11. TRANG K-MEANS
# ============================================================

elif page == "Phân cụm K-means":

    page_heading(
        "Phân cụm địa phương",
        f"Spark MLlib K-means – "
        f"Môn {subject_label}"
    )

    if (
        clusters.empty
        or cluster_selection.empty
        or cluster_eval.empty
    ):

        st.error(
            "Thiếu dữ liệu K-means cuối cùng."
        )

        st.code(str(KMEANS_DIR))
        st.stop()

    sub = clusters[
        clusters["mon_thi"] == subject
    ].copy()

    selection = cluster_selection[
        cluster_selection["mon_thi"] == subject
    ]

    if sub.empty or selection.empty:

        st.warning(
            "Môn này không đủ điều kiện "
            "để xây dựng mô hình K-means."
        )

    else:

        model = selection.iloc[0]

        k = int(model["k"])
        seed = int(model["selected_seed"])
        silhouette = float(model["silhouette"])
        ari = float(model["ari_mean"])

        a, b, c, d = st.columns(4)

        a.metric("Số cụm K", k)
        b.metric("Seed", seed)

        c.metric(
            "Silhouette trung bình",
            f"{silhouette:.4f}"
        )

        d.metric(
            "ARI trung bình",
            f"{ari:.4f}"
        )

        if ari >= 0.8:

            st.success(
                "Mô hình đạt ngưỡng ổn định "
                "ARI trung bình ≥ 0,80."
            )

        else:

            st.warning(
                "Mô hình có ARI trung bình "
                "dưới 0,80. Kết quả nhạy "
                "với việc thay đổi seed."
            )

        note(
            "K-means được huấn luyện từ "
            "10 tỷ lệ khoảng điểm đã chuẩn hóa. "
            "Biểu đồ dưới đây chỉ minh họa "
            "các cụm bằng hai chỉ số."
        )

        sub["Cụm"] = sub[
            "cluster"
        ].astype(int).map(
            lambda value: f"Cụm {value + 1}"
        )

        st.subheader(
            "Phân bố địa phương theo cụm"
        )

        fig = px.scatter(
            sub,
            x="diem_tb",
            y="distance_to_center",
            color="Cụm",
            size="so_bai",
            size_max=30,
            hover_name="province_name",
            labels={
                "diem_tb": "Điểm trung bình",
                "distance_to_center":
                    "Khoảng cách đến tâm cụm"
            },
            color_discrete_sequence=COLORS
        )

        show_chart(
            fig,
            height=460,
            legend=True
        )

        left, right = st.columns(2)

        with left:

            st.subheader(
                "Số địa phương theo cụm"
            )

            sizes = (
                sub.groupby("Cụm")
                .size()
                .reset_index(
                    name="Số địa phương"
                )
            )

            fig = px.bar(
                sizes,
                x="Cụm",
                y="Số địa phương",
                color="Cụm",
                color_discrete_sequence=COLORS
            )

            show_chart(fig, 340)

        with right:

            st.subheader(
                "Silhouette theo số cụm K"
            )

            eval_sub = cluster_eval[
                cluster_eval["mon_thi"] == subject
            ].sort_values("k")

            fig = px.line(
                eval_sub,
                x="k",
                y="silhouette",
                markers=True,
                labels={
                    "k": "Số cụm K",
                    "silhouette":
                        "Silhouette trung bình"
                }
            )

            fig.update_traces(
                line_color=COLORS[0]
            )

            show_chart(fig, 340)

        st.subheader(
            "Danh sách phân cụm"
        )

        st.dataframe(
            sub,
            use_container_width=True,
            hide_index=True
        )

        download_csv(
            sub,
            f"kmeans_{subject}.csv"
        )


# ============================================================
# 12. TRANG PHAN TICH CHUYEN SAU CA 12 MON
# ============================================================

else:

    page_heading(
        "Phân tích chuyên sâu",
        f"Phổ điểm chi tiết 0,25 và "
        f"Spike Ratio – Môn {subject_label}"
    )

    detailed_required = {
        "all_binned_national.csv":
            binned_national,
        "all_binned_province.csv":
            binned_province,
        "all_detailed_national.csv":
            exact_national,
        "all_detailed_province.csv":
            exact_province,
        "all_detailed_spikes.csv":
            detailed_spikes,
        "all_detailed_audit.csv":
            detailed_audit
    }

    missing_details = [
        filename
        for filename, data
        in detailed_required.items()
        if data.empty
    ]

    if missing_details:

        st.error(
            "Thiếu dữ liệu phân tích chuyên sâu: "
            + ", ".join(missing_details)
        )

        st.code(
            "python src\\analyze_detailed_scores.py"
        )

        st.stop()

    # Loc theo mon dang chon
    nat_b = binned_national[
        binned_national["mon_thi"] == subject
    ].copy()

    prov_b = binned_province[
        binned_province["mon_thi"] == subject
    ].copy()

    nat_e = exact_national[
        exact_national["mon_thi"] == subject
    ].copy()

    prov_e = exact_province[
        exact_province["mon_thi"] == subject
    ].copy()

    spike_s = detailed_spikes[
        detailed_spikes["mon_thi"] == subject
    ].copy()

    audit_s = detailed_audit[
        detailed_audit["mon_thi"] == subject
    ].copy()

    if (
        nat_b.empty
        or prov_b.empty
        or audit_s.empty
    ):

        st.warning(
            f"Không có dữ liệu môn {subject_label}."
        )

        st.stop()

    # --------------------------------------------------------
    # 12.1. THONG KE CHAT LUONG
    # --------------------------------------------------------

    audit = audit_s.iloc[0]

    total_valid = int(
        audit["tong_diem_hop_le"]
    )

    total_binned = int(
        nat_b["so_bai"].sum()
    )

    coverage = float(
        audit["ty_le_tren_luoi_pct"]
    )

    outside_grid = int(
        audit["so_diem_ngoai_luoi"]
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Bài thi hợp lệ",
        f"{total_valid:,}"
    )

    c2.metric(
        "Bài thi trong 40 khoảng",
        f"{total_binned:,}"
    )

    c3.metric(
        "Điểm trên lưới 0,25",
        f"{coverage:.2f}%"
    )

    c4.metric(
        "Điểm ngoài lưới",
        f"{outside_grid:,}"
    )

    if total_valid != total_binned:

        st.error(
            "Tổng số bài thi theo khoảng "
            "không khớp dữ liệu hợp lệ."
        )

        st.stop()

    note(
        "Phổ điểm theo khoảng 0,25 giữ toàn bộ "
        "điểm hợp lệ. Spike Ratio chỉ sử dụng "
        "các mức điểm chính xác trên lưới 0,25, "
        "không làm tròn điểm ngoài lưới."
    )

    # --------------------------------------------------------
    # 12.2. BO LOC DIA PHUONG
    # --------------------------------------------------------

    locations = sorted(
        prov_b["province_name"]
        .dropna()
        .unique()
        .tolist()
    )

    preferred = [
        x for x in [
            "Vĩnh Long",
            "Tây Ninh",
            "Sơn La"
        ]
        if x in locations
    ]

    default = (
        preferred
        if subject == "ngu_van"
        else locations[:3]
    )

    selected = st.multiselect(
        "Chọn địa phương phân tích chi tiết",
        locations,
        default=default,
        max_selections=6,
        key=f"deep_locations_{subject}"
    )

    left, right = st.columns(2)

    with left:

        show_national = st.checkbox(
            "Hiển thị toàn quốc",
            value=True,
            key=f"deep_national_{subject}"
        )

    with right:

        chart_mode = st.radio(
            "Dạng hiển thị",
            ["Tỷ lệ (%)", "Số bài thi"],
            horizontal=True,
            key=f"deep_mode_{subject}"
        )

    # --------------------------------------------------------
    # 12.3. BIEU DO 40 KHOANG
    # --------------------------------------------------------

    st.subheader(
        f"Phổ điểm chi tiết – {subject_label}"
    )

    labels = [
        BIN_LABELS[i]
        for i in range(40)
    ]

    fig = go.Figure()

    for i, location in enumerate(selected):

        location_data = prov_b[
            prov_b["province_name"] == location
        ]

        counts = get_bin_counts(
            location_data
        )

        if chart_mode == "Tỷ lệ (%)":

            if counts.sum() > 0:
                values = (
                    counts / counts.sum() * 100
                )
            else:
                values = counts

        else:

            values = counts

        fig.add_trace(
            go.Scatter(
                x=labels,
                y=values,
                name=location,
                mode="lines+markers",
                line=dict(
                    color=COLORS[
                        i % len(COLORS)
                    ],
                    width=2
                ),
                marker=dict(size=4)
            )
        )

    if show_national:

        counts = get_bin_counts(
            nat_b
        )

        if chart_mode == "Tỷ lệ (%)":

            values = (
                counts / counts.sum() * 100
                if counts.sum() > 0
                else counts
            )

        else:

            values = counts

        fig.add_trace(
            go.Scatter(
                x=labels,
                y=values,
                name="Toàn quốc",
                mode="lines",
                line=dict(
                    color="#27364d",
                    width=3,
                    dash="dash"
                )
            )
        )

    fig.update_layout(
        xaxis_title="Khoảng điểm rộng 0,25",
        yaxis_title=chart_mode
    )

    fig.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=labels[::4],
        ticktext=labels[::4]
    )

    show_chart(
        fig,
        height=490,
        legend=True
    )

    st.caption(
        "Mỗi bài thi được xếp vào đúng một "
        "trong 40 khoảng điểm: [0, 0,25), "
        "..., [9,75, 10]."
    )

    # --------------------------------------------------------
    # 12.4. SO SANH VOI CAC DIA PHUONG CON LAI
    # --------------------------------------------------------

    if selected:

        st.subheader(
            "Phân tích chênh lệch theo khoảng điểm"
        )

        focus = st.selectbox(
            "Địa phương cần phân tích",
            selected,
            key=f"deep_focus_{subject}"
        )

        focus_data = prov_b[
            prov_b["province_name"] == focus
        ]

        focus_code = str(
            focus_data["province_code"].iloc[0]
        )

        focus_counts = get_bin_counts(
            prov_b[
                prov_b["province_code"]
                == focus_code
            ]
        )

        rest_counts = get_bin_counts(
            prov_b[
                prov_b["province_code"]
                != focus_code
            ]
        )

        if (
            focus_counts.sum() > 0
            and rest_counts.sum() > 0
        ):

            difference = (
                focus_counts
                / focus_counts.sum()
                * 100
                -
                rest_counts
                / rest_counts.sum()
                * 100
            )

            diff_df = pd.DataFrame({
                "Khoảng điểm": labels,
                "Chênh lệch": difference
            })

            fig = px.bar(
                diff_df,
                x="Khoảng điểm",
                y="Chênh lệch",
                color_discrete_sequence=[
                    COLORS[0]
                ]
            )

            fig.add_hline(
                y=0,
                line_color="#64748b"
            )

            fig.update_yaxes(
                title="Chênh lệch (điểm phần trăm)"
            )

            fig.update_xaxes(
                type="category",
                tickmode="array",
                tickvals=labels[::4],
                ticktext=labels[::4]
            )

            show_chart(
                fig,
                height=390
            )

            max_index = int(
                np.argmax(np.abs(difference))
            )

            st.caption(
                f"Khoảng chênh lệch lớn nhất: "
                f"{labels[max_index]} – "
                f"{difference[max_index]:+.2f} "
                f"điểm phần trăm."
            )

    # --------------------------------------------------------
    # 12.5. SPIKE RATIO
    # --------------------------------------------------------

    st.subheader(
        "Phát hiện đỉnh điểm cục bộ"
    )

    if coverage < 80:

        st.warning(
            f"Chỉ {coverage:.2f}% bài thi "
            f"môn {subject_label} nằm chính xác "
            "trên lưới 0,25. Spike Ratio chỉ "
            "phản ánh tập con này."
        )

    else:

        st.info(
            f"Tỷ lệ điểm nằm chính xác trên "
            f"lưới 0,25: {coverage:.2f}%. "
            "Spike Ratio vẫn chỉ được tính "
            "từ các điểm trên lưới."
        )

    st.caption(
        "Spike Ratio tại điểm x = số bài tại x "
        "/ trung bình số bài tại x − 0,25 "
        "và x + 0,25. Giá trị cao không tự "
        "động chứng minh phân bố bất thường."
    )

    min_neighbors = st.number_input(
        "Tổng số bài tối thiểu ở hai điểm lân cận",
        min_value=2,
        max_value=50000,
        value=50,
        step=10,
        key=f"neighbors_{subject}"
    )

    spike_filtered = spike_s.copy()

    if selected:

        spike_filtered = spike_filtered[
            spike_filtered["province_name"]
            .isin(selected)
        ]

    spike_filtered = spike_filtered[
        (
            spike_filtered["so_bai_ben_trai"]
            + spike_filtered["so_bai_ben_phai"]
        ) >= min_neighbors
    ]

    spike_filtered = spike_filtered[
        spike_filtered["so_bai_tai_diem"] > 0
    ].sort_values(
        "spike_ratio",
        ascending=False
    )

    if spike_filtered.empty:

        st.info(
            "Không có mức điểm nào đáp ứng "
            "bộ lọc Spike Ratio hiện tại."
        )

    else:

        top_spikes = (
            spike_filtered.head(20).copy()
        )

        top_spikes["Nhãn"] = (
            top_spikes.apply(
                lambda row: (
                    f"{row['province_name']} – "
                    f"{row['diem']:g}"
                ),
                axis=1
            )
        )

        top_spikes = top_spikes.sort_values(
            "spike_ratio"
        )

        fig = px.bar(
            top_spikes,
            x="spike_ratio",
            y="Nhãn",
            orientation="h",
            labels={
                "spike_ratio": "Spike Ratio",
                "Nhãn": "Địa phương – điểm"
            },
            color_discrete_sequence=[
                COLORS[1]
            ]
        )

        show_chart(
            fig,
            height=490
        )

        st.subheader(
            "Bảng các mức điểm tập trung"
        )

        st.dataframe(
            spike_filtered,
            use_container_width=True,
            hide_index=True
        )

        download_csv(
            spike_filtered,
            f"spikes_{subject}.csv"
        )

    # --------------------------------------------------------
    # 12.6. NGHIEN CUU RIENG NGU VAN
    # --------------------------------------------------------

    if subject == "ngu_van":

        with st.expander(
            "Nghiên cứu tình huống: "
            "Ngữ văn tại điểm 5,0"
        ):

            five = spike_s[
                np.isclose(
                    spike_s["diem"],
                    5.0
                )
            ]

            interesting = five[
                five["province_name"].isin(
                    [
                        "Vĩnh Long",
                        "Tây Ninh",
                        "Sơn La"
                    ]
                )
            ]

            st.dataframe(
                interesting,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # 12.7. DU LIEU KIEM TRA
    # --------------------------------------------------------

    with st.expander(
        "Kiểm tra dữ liệu phân tích chuyên sâu"
    ):

        st.markdown(
            "**Đối chiếu tổng số bài thi**"
        )

        st.dataframe(
            audit_s,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "**Phổ điểm toàn quốc theo 40 khoảng**"
        )

        st.dataframe(
            nat_b.sort_values("bin_index"),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "**Điểm chính xác trên lưới 0,25**"
        )

        st.dataframe(
            nat_e.sort_values("diem"),
            use_container_width=True,
            hide_index=True
        )

        download_csv(
            nat_b,
            f"detailed_{subject}.csv"
        )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    THPT Analytics 2026 · Đồ án Nhập môn Big Data
    <div>
        PySpark · Spark MLlib · SciPy · Streamlit
    </div>
    <div>
        Khác biệt thống kê không đồng nghĩa sai phạm.
        Dữ liệu nguồn cần được kiểm chứng độc lập.
    </div>
</div>
""", unsafe_allow_html=True)
