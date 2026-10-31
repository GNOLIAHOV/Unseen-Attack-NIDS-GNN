# Nghiên cứu biểu diễn đồ thị và Graph Neural Network cho phát hiện các họ tấn công chưa xuất hiện trong huấn luyện trên dữ liệu NetFlow

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.8-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev/)
[![TypeScript 5](https://img.shields.io/badge/TypeScript-5.6-3178C6.svg)](https://www.typescriptlang.org/)

> **Mã nguồn Khóa luận Tốt nghiệp (Nhóm 23)**  
> **Đề tài:** *Nghiên cứu biểu diễn đồ thị và Graph Neural Network cho phát hiện các họ tấn công chưa xuất hiện trong huấn luyện trên dữ liệu NetFlow*

---

## 1. Giới thiệu & Bối cảnh Nghiên cứu

Các Hệ thống Phát hiện Xâm nhập Mạng (NIDS) dựa trên Học máy truyền thống thường đạt độ chính xác rất cao trên các cuộc tấn công đã biết (seen attacks), nhưng lại suy giảm hiệu năng nghiêm trọng khi đối mặt với **các họ tấn công mới chưa từng xuất hiện trong quá trình huấn luyện (unseen attack families — bài toán Nhận dạng Tập Mở / Open-Set Recognition)**. Nguyên nhân cốt lõi là do các mô hình bảng (tabular models) xử lý từng luồng mạng (flow) một cách độc lập, bỏ qua cấu trúc liên kết toàn cục giữa các máy trạm (hosts) và đưa ra dự đoán quá tự tin (overconfident predictions) trên dữ liệu ngoại lai (**Out-of-Distribution / Open-Set Unknown**).

Dự án này giải quyết bài toán trên thông qua ba trụ cột kỹ thuật chính trên bộ dữ liệu quy mô lớn **NF-UQ-NIDS-v2** (~75.87 triệu luồng NetFlow duy nhất, 284,339 nút IP):
1. **Biểu diễn Đồ thị Luồng – Máy trạm (Host-Flow Graph) & Kiến trúc Inductive Edge-Aware GATv2:** Mã hóa các địa chỉ IP thành đỉnh (nodes) với khởi tạo đặc trưng cấu trúc quy nạp 16 chiều (`Topological Node Init` không phụ thuộc định danh tĩnh `nn.Embedding`), kết hợp cơ chế truyền thông điệp chú ý động hai chiều (`GATv2Conv` tích hợp trực tiếp 41 đặc trưng cạnh NetFlow cùng `LayerNorm` và cấu trúc giải mã hai nhánh `Dual-Pathway`: nhánh Cấu trúc Đồ thị và nhánh Đặc trưng Luồng).
2. **Định lượng Bất định Nhận thức Ba Nhánh (Three-way Epistemic Uncertainty Quantification qua Monte Carlo Dropout):** Sử dụng độ lệch chuẩn dự đoán ($\text{Epistemic Std}$) qua $T=20$ lượt lấy mẫu ngẫu nhiên kết hợp độ bất đồng giữa hai nhánh (`Dual-Pathway Discrepancy`) để nhận diện dữ liệu ngoại lai và khoanh vùng cảnh báo các họ tấn công chưa từng xuất hiện.
3. **Nhận dạng Tập Mở (Mục 15.1 — Open-Set Recognition trên Đồ thị NetFlow):** Mở rộng không gian nhãn quyết định từ tập đóng nhị phân $\{0: \text{Known Benign}, 1: \text{Known Attack}\}$ sang không gian tập mở 3 lớp $\{0: \text{Known Benign}, 1: \text{Known Attack}, 2: \text{Unknown/Unseen Attack}\}$ và đánh giá toàn diện bằng các thước đo chuẩn học thuật OSR (**AUROC**, **AUPR-Unknown**, **OSCR — Open-Set Classification Rate** và **Ma trận nhầm lẫn Tập Mở 3 lớp**).

---

## 2. Cấu trúc Thư mục Dự án (Repository Structure)

Dự án được tổ chức theo chuẩn **MLOps**, tách biệt rõ ràng giữa dữ liệu, mô hình, mã nguồn lõi (`src/models/`), kịch bản huấn luyện/đánh giá và ứng dụng Web Console (`app/`):

```text
KLTN_NF-UQ-NIDS-v2/
├── app/                                  # Ứng dụng Web Console & Dịch vụ Suy luận (Production Prototype)
│   ├── api.py                            # FastAPI Backend phục vụ suy luận GATv2 + Three-way OSR & Static SPA
│   └── web/                              # Giao diện SOC Console (React 18 + TypeScript 5 + Vite, Hallmark Console)
│       ├── src/                          # Mã nguồn React (App.tsx, main.tsx, tokens.css, base.css, app.css)
│       ├── dist/                         # Bản đóng gói tĩnh phục vụ trực tiếp tại http://127.0.0.1:8000/
│       ├── index.html                    # Trang gốc tải phông chữ kỹ thuật (Space Grotesk, Inter, JetBrains Mono)
│       ├── package.json                  # Cấu hình thư viện và kịch bản build Frontend
│       ├── tsconfig.json                 # Cấu hình kiểm tra kiểu tĩnh TypeScript
│       └── vite.config.ts                # Cấu hình đóng gói Vite và proxy API phát triển
├── data/                                 # Quản lý vòng đời dữ liệu (loại trừ khỏi Git qua .gitignore)
│   ├── raw/                              # Dữ liệu gốc NF-UQ-NIDS-v2.csv (~13.7 GB) & gói nén .zip
│   ├── interim/                          # 4 tập CSV sau khi lọc trùng & chia Inductive Split (train, val, test_seen, test_unseen)
│   ├── processed/                        # Dữ liệu cạnh đồ thị (.parquet), ip_node_map, demo_microbatch & preprocessor_params.json
│   └── samples/                          # Mẫu lưu lượng mạng thực tế & file Wireshark phục vụ kiểm thử Web Console
│       ├── real_traffic_sample.csv       # 500 luồng NetFlow thực tế chưa chuẩn hóa (43 cột)
│       ├── sample_external_wireshark_log.csv # Mẫu log Wireshark 8 cột kiểm thử cơ chế tự bù khuyết trường
│       └── README.md                     # Hướng dẫn chi tiết kiểm thử mẫu dữ liệu
├── models/                               # Trọng số các mô hình đã huấn luyện
│   ├── baselines/
│   │   └── xgboost_baseline_b0.json      # Mô hình Baseline phi đồ thị (B0 - XGBoost)
│   └── gnns/
│       ├── gcn_g0_model.pt               # Mô hình G0: Transductive GCN + nn.Embedding
│       ├── gine_g1_model.pt              # Mô hình G1: Inductive GINEConv (Sum Aggregation)
│       └── gatv2_g1_v2_model.pt          # Mô hình G1_v2: Inductive Edge-Aware GATv2 + LayerNorm + Dual-Pathway
├── results/
│   └── metrics/                          # Lưu trữ kết quả thực nghiệm & đo lường hiệu năng (.json)
│       ├── baseline_b0_metrics.json      # Kết quả thực nghiệm Baseline B0 (XGBoost)
│       ├── gnn_g0_metrics.json           # Kết quả thực nghiệm G0 (Transductive GCN)
│       ├── gnn_g1_metrics.json           # Kết quả thực nghiệm G1 (Inductive GINEConv)
│       ├── gnn_g1_v2_metrics.json        # Kết quả thực nghiệm G1_v2 (Inductive GATv2 T=1)
│       ├── uq_mc_dropout_metrics.json    # Kết quả lọc báo động giả 2 nhánh MC Dropout (T=20)
│       ├── uq_three_way_metrics.json     # Kết quả quyết định 3 nhánh Three-way UQ (T=20)
│       ├── osr_open_set_metrics.json     # Kết quả Mục 15.1: Nhận dạng Tập Mở (AUROC, AUPR, OSCR, 3x3 CM)
│       └── throughput_benchmark.json     # Đo lường thông lượng (Throughput), độ trễ (Latency) & bộ nhớ
├── src/                                  # Mã nguồn cốt lõi (Core Python Modules)
│   ├── __init__.py                       # Đánh dấu mô-đun gói tin Python cấp cao
│   ├── data_prep/
│   │   ├── __init__.py
│   │   ├── eda_check.py                  # Thống kê khám phá dữ liệu bằng Polars Streaming
│   │   ├── data_split.py                 # Chia dữ liệu theo giao thức Inductive Split (Out-of-core Hash Partition)
│   │   └── preprocess_and_graph.py       # Chuẩn hóa chống rò rỉ dữ liệu & trích xuất cạnh đồ thị PyG
│   ├── models/
│   │   ├── __init__.py                   # Xuất khẩu các lớp kiến trúc GNN dùng chung
│   │   └── gnn_architectures.py          # Định nghĩa GNNBaselineG0, GNNInductiveG1, GNNInductiveGATv2 & Node Init
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_baseline_b0.py          # Huấn luyện & đánh giá Baseline B0 (XGBoost)
│   │   ├── train_gnn_g0.py               # Huấn luyện & đánh giá G0 (Transductive GCN)
│   │   ├── train_gnn_g1.py               # Huấn luyện & đánh giá G1 (Inductive GINEConv)
│   │   └── train_gnn_g1_v2.py            # Huấn luyện & đánh giá G1_v2 (Inductive GATv2 + LayerNorm)
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluate_uq_mc_dropout.py     # Đánh giá MC Dropout (T=20) với cơ chế lọc báo động giả 2 nhánh
│       ├── evaluate_uq_three_way.py      # Đánh giá quy tắc quyết định 3 nhánh (Three-way UQ Decision)
│       ├── evaluate_osr_open_set.py      # Đánh giá Mục 15.1: Open-Set Recognition (AUROC, AUPR-Out, OSCR)
│       └── eval_throughput.py            # Đo lường thông lượng (Throughput), độ trễ (Latency) & bộ nhớ
├── tools/                                # Công cụ hỗ trợ & tiện ích chuyển đổi dữ liệu
│   ├── __init__.py
│   └── pcap_to_netflow.py                # Chuyển đổi file bắt gói tin thực tế (.pcap / .pcapng) sang NetFlow CSV
├── docs/                                 # Đề cương và báo cáo khóa luận tốt nghiệp (.docx)
├── .gitignore                            # Cấu hình loại trừ các file dữ liệu, môi trường ảo và trọng số nặng
├── requirements.txt                      # Danh sách thư viện Python phụ thuộc đã tinh gọn và chuẩn hóa
└── README.md                             # Tài liệu tổng quan và hướng dẫn vận hành dự án
```

---

## 3. Kiến trúc Hệ thống & Đóng góp Thực nghiệm

### 3.1. Giao thức Chia dữ liệu Inductive Chống Rò rỉ (Unseen Attack Simulation Protocol)
- Toàn bộ **75,987,976** bản ghi ban đầu được lọc bỏ dòng trùng lặp còn **75,867,694** luồng duy nhất (trên **284,339** địa chỉ IP duy nhất).
- **Tập `Test_unseen` (Open-Set Unknown Attack — 259,458 luồng):** Tách biệt hoàn toàn 100% các bản ghi thuộc hai họ tấn công **`Bot`** (`143,097` luồng) và **`Infiltration` (`Infilteration`)** (`116,361` luồng). Hai họ tấn công này tuyệt đối không xuất hiện trong tập `Train`, `Val` hay `Test_seen`.
- **Tập `Seen` (75,608,236 luồng):** Xáo trộn ngẫu nhiên (`seed=42`) và chia theo tỷ lệ **70% Train** (`52,925,623` luồng) – **15% Validation** (`11,341,070` luồng) – **15% Test_seen** (`11,341,543` luồng).
- **Chống rò rỉ đặc trưng (Strict Anti-Leakage Preprocessing):** Bộ từ vựng của 9 cột phân loại (Categorical) và tham số $(\mu, \sigma)$ của 32 cột số thực (Numerical) được tính toán **duy nhất trên tập `Train`** trước khi biến đổi cả 4 tập.

### 3.2. Quy tắc Phân loại 3 Nhánh & Nhận dạng Tập Mở (Three-way OSR Decision Rule)
Với mỗi luồng giao tiếp $e=(u, v)$ trong một micro-batch/sliding window, mô hình `GATv2` thực hiện mã hóa đồ thị con một lần (`encode_subgraph`) và chạy $T=20$ lượt truyền thuận có bật Dropout (`MC Dropout`) trên bộ giải mã hai nhánh để ước lượng xác suất trung bình $\hat{p}$ (`Mean Prob`), độ lệch chuẩn nhận thức $\hat{\sigma}$ (`Epistemic Std`), độ bất đồng giữa hai nhánh $\Delta_{\text{path}} = |p_{\text{topo}} - p_{\text{flow}}|$, và điểm Nhận dạng Tập Mở lai ghép $S_{\text{OSR}} = \max(\hat{\sigma}, 0.35 \cdot \Delta_{\text{path}})$:
- **Nhánh 0 — `Known Benign (0)`:** $S_{\text{OSR}} \le 0.15 \land \hat{p} < 0.50 \implies$ Cho phép lưu lượng đi qua (`ALLOW`).
- **Nhánh 1 — `Known Attack (1)`:** $S_{\text{OSR}} \le 0.15 \land \hat{p} \ge 0.50 \implies$ Chặn tự động cuộc tấn công đã biết với độ tự tin cao (`BLOCK`).
- **Nhánh 2 — `Unknown / Open-Set Attack Alert (2)`:** $S_{\text{OSR}} > 0.15 \implies$ Khoanh vùng cảnh báo tấn công chưa xuất hiện / Open-Set Unknown và kích hoạt chặn/phân tích chuyên sâu (`BLOCK / SOC ALERT`).

### 3.3. Bảng Tổng hợp Kết quả Thực nghiệm Xuyên suốt Khóa luận

| Mô hình / Cơ chế | Kiến trúc & Đặc trưng Node | Precision (`Test_seen`) | Recall (`Test_seen`) | F1-score (`Test_seen`) | **Recall (`Test_unseen` Bot & Infiltration)** | **F1-score (`Test_unseen` Bot & Infiltration)** | Số mẫu Unseen bắt được (`TP / 259,458`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `B0` — XGBoost** | Cây quyết định phi đồ thị (41 cột cạnh) | `99.14%` | `98.77%` | `98.96%` | `2.13%` | `4.18%` | `5,536 / 259,458` |
| **2. `G0` — GCN Baseline** | 2 lớp `GCNConv` + `nn.Embedding(284339, 64)` | `98.99%` | `99.25%` | `99.12%` | `0.00%` | `0.00%` | `0 / 259,458` |
| **3. `G1` — GINEConv** | 2 lớp `GINEConv` (`Sum`) + `Inductive Init` | `98.26%` | `96.64%` | `97.44%` | `21.95%` | `36.00%` | `56,953 / 259,458` |
| **4. `G1_v2` — GATv2 (`T=1`)** | 2 lớp `GATv2Conv(edge_dim=41)` + `LayerNorm` | `97.56%` | `98.12%` | `97.84%` | `82.17%` | `90.21%` | `213,184 / 259,458` |
| **5. `GATv2 + 2-Branch UQ (`T=20`)`** | Lọc báo động giả 2 nhánh ($\tau_{\text{std}}=0.15$) | `99.27%` | `87.42%` | `92.97%` | `44.02%` | `61.13%` | `114,205 / 259,458` |
| **6. `GATv2 + Three-way OSR (`T=20`)`** | **Quyết định 3 nhánh Tập Mở ($T=20$, $\tau=0.15$)** | **`92.06%` (`99.27%`*)** | **`99.30%` (`99.21%`*)** | **`95.55%` (`99.24%`*)** | **`91.81%`** | **`95.73%`** | **`238,196 / 259,458`** |

> *Ghi chú (\*): Chỉ số trong ngoặc trên tập `Test_seen` là hiệu năng tại **Vùng quyết định tự tin (Nhánh 0 & Nhánh 1, $\hat{\sigma} \le 0.15$, chiếm `86.84%` tổng lưu lượng)** — giảm **73.5% báo động giả (FP từ `3,251` xuống `860` luồng)** và đạt **Precision `99.27%`**, **Recall `99.21%`**, **F1-score `99.24%`**. Trên tập `Test_unseen` (`Bot` & `Infiltration`), độ bất định nhận thức trung bình tăng vọt gấp **3.15 lần** so với `Test_seen` (`0.1441` so với `0.0458`). Cơ chế **Three-way OSR** khắc phục triệt để nghịch lý lọc bất định 2 nhánh (nâng Recall từ `44.02%` của lọc 2 nhánh và `82.17%` của `T=1` lên **`91.81%`**, F1 đạt **`95.73%`**), phát hiện **`96.72%` Bot (`138,399 / 143,097`)** và **`85.76%` Infiltration (`99,797 / 116,361`)**.*

### 3.4. Nhận dạng Tập Mở (Mục 15.1 — Open-Set Recognition Benchmark)
Đánh giá trên tập hỗn hợp Tập Mở gồm **`459,458` luồng mạng** (`200,000` luồng In-Distribution từ `Test_seen` + `259,458` luồng Open-Set Unknown từ `Test_unseen` thuộc hai họ `Bot` và `Infiltration`, kịch bản `src/evaluation/evaluate_osr_open_set.py`):

| Phương pháp Nhận dạng Tập Mở (OSR Method) | Ngưỡng ($\tau$) | **AUROC (Known vs Unknown)** | **AUPR-Unknown** | **OSCR (Open-Set Class. Rate)** | Tỷ lệ gắn cờ Nhánh 2 (`Open-Set Rate`) | **Tổng Tỷ lệ Chặn Unknown (`Nhánh 1 + 2`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Closed-Set Sigmoid Margin (`T=1`)** | `0.50` | `0.8227` | `0.7918` | `0.8151` | `16.15%` | `87.76%` |
| **2. Dual-Pathway Energy Discrepancy (`T=1`)** | `0.15` | `0.7390` | `0.7351` | `0.7307` | `14.93%` | `86.65%` |
| **3. Three-way Epistemic MC Dropout (`T=20`)** | `0.15` | `0.8531` | `0.8045` | `0.8417` | `47.79%` | `91.80%` |
| **4. Hybrid Epistemic + Dual-Pathway OSR (`T=20` — Đề xuất)** | **`0.15`** | **`0.8531`** | **`0.8044`** | **`0.8417`** | **`47.79%`** | **`91.81%`** |

- **Hiệu năng Phòng thủ Tổng thể trên toàn bộ 459,458 luồng Tập Mở (`Benign` vs `Threat {Known Attack U Unknown Attack}`):** đạt **Precision = `97.02%`**, **Recall = `94.35%`**, **F1-score = `95.66%`**.
- **Sự bổ trợ tự nhiên giữa hai nhánh của cơ chế Tập Mở:**
  - Với họ **`Bot` (`143,097` luồng):** Có cấu trúc càn quét tương đồng một phần với các cuộc tấn công đã biết nên Nhánh 1 (`Known Attack`) bắt được `96,120` luồng (`67.17%`) và Nhánh 2 (`Open-Set Unknown Alert`) bắt tiếp `42,279` luồng (`29.55%`), đạt tổng tỷ lệ chặn **`96.72%`**.
  - Với họ **`Infiltration` (`116,361` luồng):** Là tấn công thẩm thấu ngụy trang tinh vi khiến dự đoán tất định bị lưỡng lự ($\hat{\sigma}_{\text{TB}} = 0.1808$, độ lệch hai nhánh $\Delta_{\text{path, TB}} = 0.2283$), được Nhánh 2 (`Open-Set Unknown Alert`) phát hiện trực tiếp tới **`81,713` luồng (`70.22%`)**, kết hợp Nhánh 1 (`18,084` luồng) nâng tổng tỷ lệ chặn lên **`85.76%`**.

### 3.5. Hiệu năng Hệ thống (System Throughput & Latency Benchmark)
Đo lường thực tế trên 50 mini-batches liên tiếp (`204,800` luồng NetFlow, `batch_size = 4096`, `num_neighbors = [10, 10]`):

| Kịch bản vận hành | Throughput (Flows/giây) | Độ trễ trung bình / Batch (`4096` flows) | Độ trễ trung bình / Luồng | Bộ nhớ đỉnh (Peak RAM) |
| :--- | :---: | :---: | :---: | :---: |
| **Standard Inference (`T=1`, Deterministic)** | **`168,625.47` flows/s** | `24.17 ms` | `5.93 us/flow` | `535.86 MB` |
| **MC Dropout Three-way OSR (`T=20`)** | **`7,077.21` flows/s** | `578.65 ms` | `141.30 us/flow` | `540.41 MB` |

---

## 4. Hướng dẫn Cài đặt & Khởi chạy Hệ thống

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

### 4.2. Thực thi Đường ống Huấn luyện & Đánh giá Thực nghiệm
```powershell
# 1. Tiền xử lý & tạo đồ thị (nếu chạy lại từ dữ liệu thô):
.\env\Scripts\python.exe src\data_prep\data_split.py
.\env\Scripts\python.exe src\data_prep\preprocess_and_graph.py

# 2. Huấn luyện các mô hình Baseline & GNN:
.\env\Scripts\python.exe src\training\train_baseline_b0.py
.\env\Scripts\python.exe src\training\train_gnn_g0.py
.\env\Scripts\python.exe src\training\train_gnn_g1.py
.\env\Scripts\python.exe src\training\train_gnn_g1_v2.py

# 3. Đánh giá Định lượng Bất định (UQ), Nhận dạng Tập Mở (OSR) & Thông lượng:
.\env\Scripts\python.exe src\evaluation\evaluate_uq_mc_dropout.py
.\env\Scripts\python.exe src\evaluation\evaluate_uq_three_way.py
.\env\Scripts\python.exe src\evaluation\evaluate_osr_open_set.py
.\env\Scripts\python.exe src\evaluation\eval_throughput.py
```

### 4.3. Khởi chạy Ứng dụng Web Console & Backend API (1 Lệnh Duy Nhất)
Giao diện **React 18 + TypeScript Console (`app/web/`)** đã được đóng gói tĩnh vào `app/web/dist/` và tích hợp trực tiếp vào máy chủ FastAPI:

```powershell
.\env\Scripts\uvicorn.exe app.api:app --host 127.0.0.1 --port 8000
```

- **Giao diện SOC Web Console (React + TypeScript):** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Tài liệu API tương tác (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Kiểm tra trạng thái hệ thống:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Số liệu thực nghiệm JSON:** [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics)

*(Tùy chọn: Nếu chỉnh sửa mã nguồn giao diện trong `app/web/src/`, chạy `npm --prefix app/web run build` để cập nhật bản đóng gói tĩnh hoặc `npm --prefix app/web run dev` để chạy Vite Dev Server tại cổng `5173`).*

### 4.4. Thử nghiệm Dữ liệu Mạng Thực tế & Chuyển đổi Gói tin Wireshark
- **Cách 1 — Thử nghiệm trực tiếp tệp mẫu:** Mở [http://127.0.0.1:8000/](http://127.0.0.1:8000/), bấm **"Tải tệp .parquet / .csv"** và chọn `data/samples/real_traffic_sample.csv` (500 luồng NetFlow thật) hoặc `data/samples/sample_external_wireshark_log.csv` (log Wireshark 8 cột).
- **Cách 2 — Tự bắt gói tin máy tính & suy luận trực tiếp:**
  ```powershell
  # Bắt gói tin qua Wireshark và lưu dạng file.pcap hoặc file.pcapng, sau đó chuyển đổi:
  python tools/pcap_to_netflow.py duong_dan_goi_tin.pcapng data/samples/my_traffic.csv
  ```
  Tải tệp `data/samples/my_traffic.csv` lên Web Console để theo dõi mô hình phân loại lưu lượng thường (`Benign ALLOW`) và khoanh vùng cảnh báo các kết nối lạ / đột biến (`Epistemic OSR Alert`).

---

## 5. Thông tin Nhóm Thực hiện

- **Nhóm thực hiện:** Nhóm 23
- **Sinh viên thực hiện:**
  - **Nguyễn Đức Hưng Thịnh**
  - **Võ Hải Long**
- **Giảng viên hướng dẫn:** **TS. Võ Quang Hoàng Khang**
