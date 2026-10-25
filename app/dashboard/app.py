import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st


# =============================================================================
# 1. CẤU HÌNH TRANG & STYLE GIAO DIỆN (STREAMLIT CONFIG)
# =============================================================================
st.set_page_config(
    page_title="NIDS Dashboard: Unseen Attack Detection via Epistemic Uncertainty",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .kpi-card {
        border-radius: 10px;
        padding: 16px 20px;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12);
        text-align: center;
        margin-bottom: 10px;
    }
    .kpi-title {
        font-size: 0.92rem;
        opacity: 0.92;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
    }
    .kpi-sub {
        font-size: 0.80rem;
        opacity: 0.85;
        margin-top: 4px;
    }
    .bg-total { background: linear-gradient(135deg, #1e293b, #334155); }
    .bg-latency { background: linear-gradient(135deg, #2563eb, #1d4ed8); }
    .bg-benign { background: linear-gradient(135deg, #16a34a, #15803d); }
    .bg-attack { background: linear-gradient(135deg, #ea580c, #c2410c); }
    .bg-zeroday { background: linear-gradient(135deg, #dc2626, #991b1b); }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 2. HÀM GỌI FASTAPI & TRỰC QUAN HÓA
# =============================================================================
def call_predict_file_api(
    api_base_url: str,
    file_bytes: bytes,
    filename: str,
    max_rows: int,
    mc_samples: int,
    std_threshold: float,
) -> dict:
    """Gửi file dạng multipart/form-data tới endpoint /predict_file của FastAPI Backend."""
    url = f"{api_base_url.rstrip('/')}/predict_file"
    params = {
        "max_rows": max_rows,
        "mc_samples": mc_samples,
        "std_threshold": std_threshold,
    }
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    response = requests.post(url, params=params, files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def render_uncertainty_scatter_plot(df_pred: pd.DataFrame, std_threshold: float):
    """
    Vẽ biểu đồ Scatter Plot:
    - Trục X: Mean Probability (0.0 -> 1.0)
    - Trục Y: Epistemic Uncertainty (Std)
    - Đường gạch ngang tại ngưỡng Std = std_threshold (mặc định 0.15)
    - Đường gạch dọc tại ngưỡng Mean Prob = 0.50
    - Tô màu 3 nhánh quyết định:
        + 0: Benign (Trusted) -> Xanh lá (#16a34a)
        + 1: Known Attack     -> Cam (#ea580c)
        + 2: Zero-day Alert   -> Đỏ (#dc2626)
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=130)

    palette = {
        "Benign (Trusted)": "#16a34a",
        "Known Attack": "#ea580c",
        "Unseen Attack Alert (High Epistemic Uncertainty)": "#dc2626",
    }
    markers = {
        "Benign (Trusted)": "o",
        "Known Attack": "s",
        "Unseen Attack Alert (High Epistemic Uncertainty)": "^",
    }

    y_max = max(0.45, float(df_pred["epistemic_std"].max()) * 1.15 if len(df_pred) > 0 else 0.45)
    ax.axhspan(std_threshold, y_max, color="#fee2e2", alpha=0.38, zorder=0)
    ax.fill_between([0.0, 0.5], 0.0, std_threshold, color="#dcfce7", alpha=0.35, zorder=0)
    ax.fill_between([0.5, 1.0], 0.0, std_threshold, color="#ffedd5", alpha=0.35, zorder=0)

    sns.scatterplot(
        data=df_pred,
        x="mean_prob",
        y="epistemic_std",
        hue="predicted_class",
        style="predicted_class",
        palette=palette,
        markers=markers,
        s=75,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
        zorder=3,
    )

    ax.axhline(
        y=std_threshold,
        color="#b91c1c",
        linestyle="--",
        linewidth=2.0,
        label=f"Uncertainty Threshold (Std = {std_threshold:.2f})",
        zorder=4,
    )
    ax.axvline(
        x=0.50,
        color="#475569",
        linestyle=":",
        linewidth=1.5,
        label="Probability Threshold (p = 0.50)",
        zorder=4,
    )

    ax.text(
        0.02,
        y_max * 0.90,
        "VÙNG CẢNH BÁO UNSEEN ATTACK / OOD (Nhánh 3: Std > Threshold)\nCác mẫu Unseen Attack lơ lửng ở vùng bất định nhận thức cao",
        fontsize=9.5,
        fontweight="bold",
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#dc2626", alpha=0.85),
    )
    ax.text(
        0.03,
        std_threshold * 0.18,
        "Nhánh 1: Trusted Benign\n(Mean < 0.5, Std <= Threshold)",
        fontsize=8.5,
        fontweight="bold",
        color="#166534",
    )
    ax.text(
        0.62,
        std_threshold * 0.18,
        "Nhánh 2: Known Attack\n(Mean >= 0.5, Std <= Threshold)",
        fontsize=8.5,
        fontweight="bold",
        color="#9a3412",
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("Mean Predicted Probability (MC Dropout T=20)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Epistemic Uncertainty (Standard Deviation)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Không gian Quyết định 3 Nhánh (Three-way UQ Decision Space: Mean Prob vs Epistemic Std)",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9)
    fig.tight_layout()
    return fig


def highlight_zeroday_rows(row: pd.Series):
    """Bôi đỏ nổi bật các dòng được phát hiện là Unseen Attack Alert và tô cam nhẹ Known Attack."""
    pred_class = str(row.get("predicted_class", ""))
    if "Unseen Attack Alert" in pred_class:
        return ["background-color: #fee2e2; color: #991b1b; font-weight: 600;"] * len(row)
    if "Known Attack" in pred_class:
        return ["background-color: #ffedd5; color: #9a3412;"] * len(row)
    return [""] * len(row)


# =============================================================================
# 3. GIAO DIỆN CHÍNH (MAIN UI LAYOUT)
# =============================================================================
def main():
    st.title("NIDS Dashboard: Unseen Attack Detection via Epistemic Uncertainty")
    st.caption(
        "Hệ thống Phát hiện Xâm nhập Mạng dựa trên Đồ thị **Inductive Edge-Aware GATv2** "
        "kết hợp **Định lượng Bất định Nhận thức 3 Nhánh (Three-way Monte Carlo Dropout UQ)**"
    )

    # -------------------------------------------------------------------------
    # SIDEBAR: CẤU HÌNH KẾT NỐI API & UPLOAD MICRO-BATCH
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("Cấu hình & Đầu vào")
        api_base_url = st.text_input(
            "FastAPI Backend URL",
            value="http://localhost:8000",
            help="Địa chỉ máy chủ FastAPI đang chạy endpoint /predict_file",
        )

        backend_online = False
        try:
            health_resp = requests.get(f"{api_base_url.rstrip('/')}/health", timeout=3)
            if health_resp.status_code == 200:
                backend_online = True
                st.success("FastAPI Backend: Đang hoạt động")
            else:
                st.warning(f"FastAPI phản hồi mã: {health_resp.status_code}")
        except Exception:
            st.error("Chưa kết nối được FastAPI (`http://localhost:8000`)")

        st.divider()
        st.subheader("Tải lên Micro-batch NetFlow")
        uploaded_file = st.file_uploader(
            "Chọn file dữ liệu luồng mạng (.csv hoặc .parquet)",
            type=["csv", "parquet"],
            help="File chứa 41 cột đặc trưng NetFlow đã tiền xử lý (ví dụ từ data/processed/)",
        )

        max_rows = st.slider("Số luồng tối đa phân tích (Max Rows)", min_value=20, max_value=2000, value=250, step=10)
        mc_samples = st.slider("Số lần lấy mẫu MC Dropout (T)", min_value=5, max_value=50, value=20, step=5)
        std_threshold = st.slider(
            "Ngưỡng Bất định Unseen Attack (Std Threshold)",
            min_value=0.05,
            max_value=0.35,
            value=0.15,
            step=0.01,
        )

    # -------------------------------------------------------------------------
    # XỬ LÝ GỌI API KHI NGƯỜI DÙNG UPLOAD FILE
    # -------------------------------------------------------------------------
    if uploaded_file is None:
        st.info(
            "Hãy tải lên một file `.csv` hoặc `.parquet` ở thanh Sidebar bên trái "
            "(ví dụ: `data/processed/edges_test_unseen.parquet` hoặc `data/processed/edges_test_seen.parquet`) "
            "để bắt đầu phân tích!"
        )
        return

    if not backend_online:
        st.error(
            "Không thể gửi yêu cầu phân tích vì FastAPI Backend chưa chạy ở cổng 8000. "
            "Vui lòng bật `uvicorn app.api:app --port 8000`."
        )
        return

    with st.spinner(
        f"Đang gửi '{uploaded_file.name}' tới FastAPI (`POST /predict_file`, MC Dropout T={mc_samples})..."
    ):
        try:
            res = call_predict_file_api(
                api_base_url=api_base_url,
                file_bytes=uploaded_file.getvalue(),
                filename=uploaded_file.name,
                max_rows=max_rows,
                mc_samples=mc_samples,
                std_threshold=std_threshold,
            )
        except Exception as exc:
            st.error(f"Lỗi khi gọi API `/predict_file`: {exc}")
            return

    predictions = res.get("predictions", [])
    counts = res.get("summary_counts", {})
    total_flows = res.get("total_flows", len(predictions))
    latency_ms = res.get("latency_ms", 0.0)
    used_std_threshold = res.get("std_threshold", std_threshold)

    n_benign = counts.get("benign_0", 0)
    n_known = counts.get("known_attack_1", 0)
    n_zeroday = counts.get("zeroday_alert_2", 0)

    # =========================================================================
    # HÀNG 1: KPI METRICS (TỔNG SỐ LUỒNG, LATENCY, BENIGN, KNOWN ATTACK, ZERO-DAY)
    # =========================================================================
    st.subheader("Hàng 1: Chỉ số Giám sát Thời gian thực (KPI Metrics)")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card bg-total">
                <div class="kpi-title">Tổng số Luồng</div>
                <div class="kpi-value">{total_flows:,}</div>
                <div class="kpi-sub">File: {res.get('filename', 'N/A')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="kpi-card bg-latency">
                <div class="kpi-title">Độ trễ (Latency)</div>
                <div class="kpi-value">{latency_ms:,.1f} ms</div>
                <div class="kpi-sub">MC Dropout T={res.get('mc_samples_T', mc_samples)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        pct_b = (n_benign / total_flows * 100) if total_flows else 0.0
        st.markdown(
            f"""
            <div class="kpi-card bg-benign">
                <div class="kpi-title">Benign (Trusted)</div>
                <div class="kpi-value">{n_benign:,}</div>
                <div class="kpi-sub">{pct_b:.1f}% tổng lưu lượng</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        pct_k = (n_known / total_flows * 100) if total_flows else 0.0
        st.markdown(
            f"""
            <div class="kpi-card bg-attack">
                <div class="kpi-title">Known Attack</div>
                <div class="kpi-value">{n_known:,}</div>
                <div class="kpi-sub">{pct_k:.1f}% (Mean >= 0.5, Std <= {used_std_threshold})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col5:
        pct_z = (n_zeroday / total_flows * 100) if total_flows else 0.0
        st.markdown(
            f"""
            <div class="kpi-card bg-zeroday">
                <div class="kpi-title">Unseen Attack Alert</div>
                <div class="kpi-value">{n_zeroday:,}</div>
                <div class="kpi-sub">{pct_z:.1f}% (Epistemic Std > {used_std_threshold})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    df_pred = pd.DataFrame(predictions)
    if df_pred.empty:
        st.warning("Không có dòng dự đoán nào được trả về.")
        return

    # =========================================================================
    # HÀNG 2: BIỂU ĐỒ PHÂN TÁN (SCATTER PLOT: MEAN PROB vs EPISTEMIC STD)
    # =========================================================================
    st.subheader("Hàng 2: Không gian Phân bố Bất định Nhận thức (Mean Prob vs. Epistemic Std)")
    fig = render_uncertainty_scatter_plot(df_pred, used_std_threshold)
    st.pyplot(fig, use_container_width=True)

    # =========================================================================
    # HÀNG 3: BẢNG CHI TIẾT DỰ ĐOÁN (DATA TABLE - BÔI ĐỎ ZERO-DAY ALERT)
    # =========================================================================
    st.subheader("Hàng 3: Bảng Chi tiết Kết quả Dự đoán từng Luồng (Flow-level Predictions)")

    filter_col1, _ = st.columns([2, 3])
    with filter_col1:
        selected_classes = st.multiselect(
            "Lọc theo Nhánh Quyết định:",
            options=list(df_pred["predicted_class"].unique()),
            default=list(df_pred["predicted_class"].unique()),
        )

    df_view = df_pred[df_pred["predicted_class"].isin(selected_classes)].copy()

    display_cols = [
        "flow_index",
        "src_node",
        "dst_node",
        "mean_prob",
        "epistemic_std",
        "predicted_label",
        "predicted_class",
        "action",
    ]
    if "ground_truth_attack" in df_view.columns:
        display_cols.append("ground_truth_attack")
    if "ground_truth_label" in df_view.columns:
        display_cols.append("ground_truth_label")

    df_view = df_view[[c for c in display_cols if c in df_view.columns]]

    styled_df = df_view.style.apply(highlight_zeroday_rows, axis=1).format(
        {
            "mean_prob": "{:.4f}",
            "epistemic_std": "{:.4f}",
        }
    )
    st.dataframe(styled_df, use_container_width=True, height=420)


if __name__ == "__main__":
    main()
