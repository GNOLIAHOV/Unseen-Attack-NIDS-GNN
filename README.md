# Nghiên cứu biểu diễn đồ thị và Graph Neural Network cho phát hiện các họ tấn công chưa xuất hiện trong huấn luyện trên dữ liệu NetFlow

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.8-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-FF4B4B.svg)](https://streamlit.io/)

> **Mã nguồn Khóa luận Tốt nghiệp (Nhóm 23)**  
> **Đề tài:** *Nghiên cứu biểu diễn đồ thị và Graph Neural Network cho phát hiện các họ tấn công chưa xuất hiện trong huấn luyện trên dữ liệu NetFlow*

---

## 1. Giới thiệu & Bối cảnh Nghiên cứu

Các Hệ thống Phát hiện Xâm nhập Mạng (NIDS) dựa trên Học máy truyền thống thường đạt độ chính xác rất cao trên các cuộc tấn công đã biết (seen attacks), nhưng lại suy giảm hiệu năng nghiêm trọng khi đối mặt với **các họ tấn công mới chưa từng xuất hiện trong quá trình huấn luyện (unseen attack families — mô phỏng tấn công zero-day)**. Nguyên nhân cốt lõi là do các mô hình bảng (tabular models) xử lý từng luồng mạng (flow) một cách độc lập, bỏ qua cấu trúc liên kết toàn cục giữa các máy trạm (hosts) và đưa ra dự đoán quá tự tin (overconfident predictions) trên dữ liệu ngoại lai (**Out-of-Distribution - OOD**).

Dự án này giải quyết bài toán trên thông qua hai trụ cột kỹ thuật chính trên bộ dữ liệu quy mô lớn **NF-UQ-NIDS-v2** (~75.87 triệu luồng NetFlow duy nhất, 284,339 nút IP):
1. **Biểu diễn Đồ thị Luồng – Máy trạm (Host-Flow Graph) & Kiến trúc Inductive Edge-Aware GATv2:** Mã hóa các địa chỉ IP thành đỉnh (nodes) với khởi tạo hằng số không phụ thuộc định danh tĩnh (`torch.ones`), kết hợp cơ chế truyền thông điệp chú ý động hai chiều (`GATv2Conv` tích hợp trực tiếp 41 đặc trưng cạnh NetFlow cùng `LayerNorm` và cân bằng trọng số lớp).
2. **Định lượng Bất định Nhận thức Ba Nhánh (Three-way Epistemic Uncertainty Quantification qua Monte Carlo Dropout):** Sử dụng độ lệch chuẩn dự đoán ($\text{Epistemic Std}$) qua $T=20$ lượt lấy mẫu ngẫu nhiên để nhận diện dữ liệu OOD và khoanh vùng cảnh báo các họ tấn công chưa từng xuất hiện.

---

## 2. Cấu trúc Thư mục Dự án (Repository Structure)

Dự án được tổ chức theo chuẩn **MLOps**, tách biệt rõ ràng giữa dữ liệu, mô hình, mã nguồn huấn luyện/đánh giá và ứng dụng nguyên mẫu:

```text
KLTN_NF-UQ-NIDS-v2/
├── app/                                  # Nguyên mẫu ứng dụng triển khai (Prototype Application)
│   ├── api.py                            # FastAPI Backend phục vụ suy luận GATv2 + Three-way UQ
│   └── dashboard/
│       └── app.py                        # Giao diện giám sát Streamlit NIDS Dashboard
├── data/                                 # Quản lý vòng đời dữ liệu (bị chặn bởi .gitignore)
│   ├── raw/                              # Dữ liệu gốc NF-UQ-NIDS-v2.csv (~13.7 GB) & gói .zip
│   ├── interim/                          # 4 tập CSV sau khi lọc trùng & chia Inductive Split
│   └── processed/                        # Dữ liệu cạnh đồ thị (.parquet), ip_node_map & preprocessor_params.json
├── models/                               # Trọng số các mô hình đã huấn luyện
│   ├── baselines/
│   │   └── xgboost_baseline_b0.json      # Mô hình Baseline phi đồ thị (B0 - XGBoost)
│   └── gnns/
│       ├── gcn_g0_model.pt               # Mô hình G0: Transductive GCN + nn.Embedding
│       ├── gine_g1_model.pt              # Mô hình G1: Inductive GINEConv (Sum Aggregation)
│       └── gatv2_g1_v2_model.pt          # Mô hình G1_v2: Inductive Edge-Aware GATv2 + LayerNorm
├── results/
│   └── metrics/                          # Lưu trữ kết quả thực nghiệm & đo lường hiệu năng (.json)
│       ├── baseline_b0_metrics.json
│       ├── gnn_g0_metrics.json
│       ├── gnn_g1_metrics.json
│       ├── gnn_g1_v2_metrics.json
│       ├── uq_mc_dropout_metrics.json
│       ├── uq_three_way_metrics.json
│       └── throughput_benchmark.json
├── src/                                  # Mã nguồn cốt lõi (Core Source Code)
│   ├── data_prep/
│   │   ├── eda_check.py                  # Thống kê khám phá dữ liệu bằng Polars Streaming
│   │   ├── data_split.py                 # Chia dữ liệu theo giao thức Inductive Split (Out-of-core Hash Partition)
│   │   └── preprocess_and_graph.py       # Chuẩn hóa chống rò rỉ dữ liệu & trích xuất cạnh đồ thị PyG
│   ├── training/
│   │   ├── train_baseline_b0.py          # Huấn luyện & đánh giá Baseline B0 (XGBoost)
│   │   ├── train_gnn_g0.py               # Kiểm thử đường ống mini-batch LinkNeighborLoader cho GCN
│   │   ├── train_gnn_g0_full.py          # Huấn luyện & đánh giá G0 (Transductive GCN)
│   │   ├── train_gnn_g1.py               # Huấn luyện & đánh giá G1 (Inductive GINEConv)
│   │   └── train_gnn_g1_v2.py            # Huấn luyện & đánh giá G1_v2 (Inductive GATv2 + LayerNorm)
│   ├── evaluation/
│   │   ├── evaluate_uq_mc_dropout.py     # Đánh giá MC Dropout (T=20) với cơ chế lọc báo động giả
│   │   ├── evaluate_uq_three_way.py      # Đánh giá quy tắc quyết định 3 nhánh (Three-way UQ Decision)
│   │   └── eval_throughput.py            # Đo lường thông lượng (Throughput), độ trễ (Latency) & bộ nhớ
│   └── models/                           # Thư mục mở rộng cho các mô-đun kiến trúc GNN
├── docs/                                 # Tài liệu kỹ thuật và báo cáo khóa luận
├── .gitignore                            # Cấu hình loại trừ các file dữ liệu và trọng số nặng
├── requirements.txt                      # Danh sách thư viện và phiên bản phụ thuộc
└── setup_workspace.py                    # Script tự động chuẩn hóa cấu trúc thư mục MLOps
```

---

## 3. Kiến trúc Hệ thống & Đóng góp Thực nghiệm

### 3.1. Giao thức Chia dữ liệu Inductive Chống Rò rỉ (Zero-day Simulation Protocol)
- Toàn bộ **75,987,976** bản ghi ban đầu được lọc bỏ dòng trùng lặp còn **75,867,694** luồng duy nhất (trên **284,339** địa chỉ IP duy nhất).
- **Tập `Test_unseen` (Zero-day — 53,861 luồng):** Tách biệt hoàn toàn 100% các bản ghi thuộc hai họ tấn công **`Exploits`** (31,551 luồng) và **`Fuzzers`** (22,310 luồng). Hai họ tấn công này tuyệt đối không xuất hiện trong tập `Train`, `Val` hay `Test_seen`.
- **Tập `Seen` (75,813,833 luồng):** Xáo trộn ngẫu nhiên (`seed=42`) và chia theo tỷ lệ **70% Train** (`53,069,683` luồng) – **15% Validation** (`11,372,074` luồng) – **15% Test_seen** (`11,372,076` luồng).
- **Chống rò rỉ đặc trưng (Strict Anti-Leakage Preprocessing):** Bộ từ vựng của 9 cột phân loại (Categorical) và tham số $(\mu, \sigma)$ của 32 cột số thực (Numerical) được tính toán **duy nhất trên tập `Train`** trước khi biến đổi cả 4 tập.

### 3.2. Quy tắc Phân loại 3 Nhánh (Three-way UQ Decision Rule)
Với mỗi luồng giao tiếp $e=(u, v)$ trong một micro-batch/sliding window, mô hình `GATv2` thực hiện $T=20$ lượt truyền thuận có bật Dropout (`MC Dropout`) để ước lượng xác suất trung bình $\hat{p}$ (`Mean Prob`) và độ lệch chuẩn nhận thức $\hat{\sigma}$ (`Epistemic Std`):
- **Nhánh 1 — `Trusted Benign (0)`:** $\hat{p} < 0.50 \land \hat{\sigma} \le 0.15 \implies$ Cho phép lưu lượng đi qua (`ALLOW`).
- **Nhánh 2 — `Known Attack (1)`:** $\hat{p} \ge 0.50 \land \hat{\sigma} \le 0.15 \implies$ Chặn tự động cuộc tấn công đã biết với độ tự tin cao (`BLOCK`).
- **Nhánh 3 — `Zero-day / OOD Alert (2)`:** $\hat{\sigma} > 0.15 \implies$ Khoanh vùng cảnh báo tấn công chưa xuất hiện / OOD và kích hoạt chặn/phân tích chuyên sâu (`BLOCK / SOC ALERT`).

### 3.3. Bảng Tổng hợp Kết quả Thực nghiệm Xuyên suốt Khóa luận

| Mô hình / Cơ chế | Kiến trúc & Đặc trưng Node | Precision (`Test_seen`) | Recall (`Test_seen`) | F1-score (`Test_seen`) | **Recall (`Test_unseen` Zero-day)** | **F1-score (`Test_unseen` Zero-day)** | Số mẫu Zero-day bắt được (`TP / 53,861`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `B0` — XGBoost** | Cây quyết định phi đồ thị (41 cột cạnh) | `99.19%` | `98.54%` | `98.86%` | `45.71%` | `62.74%` | `24,620 / 53,861` |
| **2. `G0` — GCN Baseline** | 2 lớp `GCNConv` + `nn.Embedding(284339, 64)` | `99.09%` | `98.90%` | `99.00%` | `19.90%` | `33.20%` | `10,719 / 53,861` |
| **3. `G1` — GINEConv** | 2 lớp `GINEConv` (`Sum`) + `torch.ones` | `97.93%` | `96.74%` | `97.33%` | `5.84%` | `11.03%` | `3,145 / 53,861` |
| **4. `G1_v2` — GATv2 (T=1)** | 2 lớp `GATv2Conv(edge_dim=41)` + `LayerNorm` | `77.80%` | `99.52%` | `87.33%` | **`75.84%`** | **`86.26%`** | `40,846 / 53,861` |
| **5. `GATv2 + Three-way UQ (T=20)`** | **MC Dropout ($T=20$, $\tau_{\text{std}}=0.15$)** | **`79.40%` (`95.55%`*)** | **`99.48%` (`99.26%`*)** | **`88.31%` (`97.37%`*)** | **`82.26%`** | **`90.27%`** | **`44,308 / 53,861`** |

> *Ghi chú (\*): Chỉ số trong ngoặc trên tập `Test_seen` là hiệu năng tại **Vùng quyết định tự tin (Nhánh 1 & Nhánh 2, $\hat{\sigma} \le 0.15$)** — giảm **88.5% báo động giả (FP)** và đạt **Precision `95.55%`**, **F1-score `97.37%`**. Trên tập `Test_unseen`, riêng **Nhánh 3 ($\hat{\sigma} > 0.15$)** đã bắt gọn **`44,280 / 53,861` mẫu Zero-day (`82.94%` Exploits và `81.31%` Fuzzers)**.*

### 3.4. Hiệu năng Hệ thống (System Throughput & Latency Benchmark)
Đo lường thực tế trên 50 mini-batches liên tiếp (`204,800` luồng NetFlow, `batch_size = 4096`, `num_neighbors = [10, 10]`):

| Kịch bản vận hành | Throughput (Flows/giây) | Độ trễ trung bình / Batch (`4096` flows) | Độ trễ trung bình / Luồng | Bộ nhớ đỉnh (Peak RAM) |
| :--- | :---: | :---: | :---: | :---: |
| **Standard Inference (`T=1`, Deterministic)** | **`167,927.50` flows/s** | `24.25 ms` | `5.95 µs/flow` | `496.73 MB` |
| **MC Dropout Three-way UQ (`T=20`)** | **`7,835.71` flows/s** | `522.61 ms` | `127.62 µs/flow` | `500.67 MB` |

---

## 4. Hướng dẫn Cài đặt & Khởi chạy Demo

### 4.1. Cài đặt Môi trường
Yêu cầu hệ thống: **Python 3.10+**. Khuyến nghị kích hoạt môi trường ảo (`virtual environment`) và cài đặt các thư viện từ `requirements.txt`:

```powershell
# 1. Tạo và kích hoạt môi trường ảo (Windows PowerShell)
python -m venv env
.\env\Scripts\Activate.ps1

# 2. Cài đặt đầy đủ các gói thư viện phụ thuộc
pip install -r requirements.txt
pip install pyg_lib -f https://data.pyg.org/whl/torch-2.14.0+cpu.html
```

### 4.2. Khởi chạy Nguyên mẫu Hệ thống (FastAPI Backend & Streamlit Dashboard)

Mở **2 cửa sổ Terminal** tại thư mục gốc của dự án (đã kích hoạt `env`):

- **Terminal 1 — Khởi chạy FastAPI Backend Server (Cổng `8000`):**
  ```powershell
  uvicorn app.api:app --reload --port 8000
  ```
  - Tài liệu API tương tác (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)
  - Kiểm tra trạng thái hệ thống: [http://localhost:8000/health](http://localhost:8000/health)

- **Terminal 2 — Khởi chạy Giao diện Streamlit Dashboard (Cổng `8501`):**
  ```powershell
  streamlit run app/dashboard/app.py
  ```
  - Truy cập Dashboard trên trình duyệt tại: **[http://localhost:8501](http://localhost:8501)**
  - Tại thanh **Sidebar**, tải lên một file micro-batch định dạng `.parquet` hoặc `.csv` (ví dụ: `data/processed/edges_test_unseen.parquet` hoặc `data/processed/demo_microbatch.csv`) để quan sát trực tiếp thẻ KPI, biểu đồ phân tán **Mean Prob vs. Epistemic Std** và bảng cảnh báo **Zero-day Alert**.

---

## 5. Thông tin Nhóm Thực hiện

- **Nhóm thực hiện:** Nhóm 23
- **Sinh viên thực hiện:**
  - **Nguyễn Đức Hưng Thịnh**
  - **Võ Hải Long**
- **Giảng viên hướng dẫn:** **TS. Võ Quang Hoàng Khang**
