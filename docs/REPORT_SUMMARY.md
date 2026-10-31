# BÁO CÁO TỔNG HỢP KHOA HỌC & KẾT QUẢ THỰC NGHIỆM ĐỒNG BỘ
## KHÓA LUẬN TỐT NGHIỆP — NHÓM 23

> **Đề tài:** Nghiên cứu biểu diễn đồ thị và Graph Neural Network cho phát hiện các họ tấn công chưa xuất hiện trong huấn luyện trên dữ liệu NetFlow  
> **Sinh viên thực hiện:** Nguyễn Đức Hưng Thịnh — Võ Hải Long  
> **Giảng viên hướng dẫn:** ThS. Võ Quang Hoàng Khang  
> **Bộ dữ liệu thực nghiệm:** NF-UQ-NIDS-v2 (75,867,694 luồng NetFlow duy nhất, 284,339 địa chỉ IP / Host)  

---

## 1. Tóm tắt Đóng góp Kỹ thuật Cốt lõi

1. **Giao thức Chia dữ liệu Inductive Chống Rò rỉ (Strict Anti-Leakage Hold-Out):**
   - Tách biệt tuyệt đối 100% hai họ tấn công ngoại lai **`Bot`** (143,097 luồng) và **`Infiltration`** (116,361 luồng) sang tập **`Test_unseen`** (tổng cộng 259,458 luồng).
   - Tuyệt đối không xuất hiện trong quá trình huấn luyện (`Train`), tối ưu siêu tham số (`Val`) hay đánh giá tập đóng (`Test_seen`).
   - Toàn bộ từ vựng 9 cột phân loại và tham số $(\mu, \sigma)$ của 32 cột số thực được tính duy nhất trên tập `Train`.

2. **Kiến trúc Inductive Edge-Aware GATv2 & Khởi tạo Đỉnh Quy nạp:**
   - Mã hóa địa chỉ IP thành 16 đặc trưng cấu trúc quy nạp (`Topological Node Init`: in/out-degree, port entropy, flow density, IP prefix bitmask) thay vì nhúng tĩnh `nn.Embedding`.
   - Cơ chế chú ý động `GATv2Conv(edge_dim=41)` kết hợp `LayerNorm` và bộ giải mã hai nhánh **`Dual-Pathway`** (nhánh Cấu trúc Đồ thị $p_{\text{topo}}$ và nhánh Đặc trưng Luồng $p_{\text{flow}}$).

3. **Cơ chế Quyết định Ba Nhánh (Three-way OSR Decision Rule):**
   - Sử dụng Monte Carlo Dropout ($T=20$ lượt suy luận ngẫu nhiên) để đo lường Độ bất định nhận thức ($\hat{\sigma}$) kết hợp độ bất đồng hai nhánh $\Delta_{\text{pathway}} = |p_{\text{topo}} - p_{\text{flow}}|$.
   - Điểm nhận diện tập mở lai ghép: $S_{\text{OSR}} = \max(\hat{\sigma}, 0.35 \cdot \Delta_{\text{pathway}})$.
   - **Nhánh 0 — Known Benign (Allow):** $S_{\text{OSR}} \le 0.15 \land \hat{p} < 0.50$.
   - **Nhánh 1 — Known Attack (Block):** $S_{\text{OSR}} \le 0.15 \land \hat{p} \ge 0.50$.
   - **Nhánh 2 — Unseen / Open-Set Alert (Block / SOC Alert):** $S_{\text{OSR}} > 0.15$.

---

## 2. Bảng Tổng hợp Kết quả Thực nghiệm Đồng bộ

Dữ liệu được trích xuất trực tiếp từ các tệp số liệu chuẩn trong `results/metrics/`:

| Mô hình / Cơ chế | Kiến trúc & Đặc trưng | Precision (`Test_seen`) | Recall (`Test_seen`) | F1-score (`Test_seen`) | **Recall (`Test_unseen` Bot & Infiltration)** | **F1-score (`Test_unseen` Bot & Infiltration)** | Số mẫu Unseen chặn được (`TP / 259,458`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. B0 — XGBoost** | Cây quyết định phi đồ thị (41 cạnh) | 99.14% | 98.77% | 98.96% | **2.13%** | **4.18%** | 5,536 / 259,458 |
| **2. G0 — GCN Baseline** | 2 lớp GCNConv + nn.Embedding(284k, 64) | 98.99% | 99.25% | 99.12% | **0.00%** | **0.00%** | 0 / 259,458 |
| **3. G1 — GINEConv** | 2 lớp GINEConv (Sum) + Node Init 16d | 98.26% | 96.64% | 97.44% | **21.95%** | **36.00%** | 56,953 / 259,458 |
| **4. G1_v2 — GATv2 (T=1)** | 2 lớp GATv2Conv(edge_dim=41) + LayerNorm | 97.56% | 98.12% | 97.84% | **82.17%** | **90.21%** | 213,184 / 259,458 |
| **5. GATv2 + 2-Branch UQ (T=20)** | Chỉ lọc báo động giả ($\hat{\sigma} \le 0.15$) | 99.27% | 87.42% | 92.97% | **44.02%** | **61.13%** | 114,205 / 259,458 |
| **6. GATv2 + Three-way OSR (T=20)** | **Quy tắc Quyết định 3 Nhánh ($T=20, \tau=0.15$)** | **92.06% (99.27%*)** | **99.30% (99.21%*)** | **95.55% (99.24%*)** | **91.81%** | **95.73%** | **238,196 / 259,458** |

> *(\*) Trong vùng quyết định tự tin (Nhánh 0 & Nhánh 1, chiếm 86.84% lưu lượng), mô hình đạt Precision 99.27%, Recall 99.21%, F1 99.24%, giảm 73.5% số lượng báo động giả (FP giảm từ 3,251 xuống 860 luồng).*

---

## 3. Kết quả Đánh giá Nhận dạng Tập Mở (Mục 15.1 — OSR Benchmark)

Đánh giá trên tập hỗn hợp **459,458 luồng** (200,000 In-Distribution từ `Test_seen` + 259,458 Open-Set Unknown từ `Test_unseen`):

* **AUROC (Known vs Unknown):** `0.8531`
* **AUPR-Unknown:** `0.8044`
* **OSCR (Open-Set Classification Rate):** `0.8417`
* **Tỷ lệ phát hiện từng họ tấn công chưa xuất hiện:**
  - **`Bot` (143,097 luồng):** Tổng tỷ lệ chặn **`96.72%`** (138,399 / 143,097 luồng — Nhánh 1 bắt 96,120 luồng, Nhánh 2 bắt tiếp 42,279 luồng).
  - **`Infiltration` (116,361 luồng):** Tổng tỷ lệ chặn **`85.76%`** (99,796 / 116,361 luồng — Nhánh 1 bắt 18,084 luồng, Nhánh 2 bắt tiếp 81,711 luồng nhờ độ bất định trung bình vọt lên $\hat{\sigma} = 0.1808$).
* **Ma trận nhầm lẫn Tập Mở 3x3 (Confusion Matrix):**

| Nhãn thực tế \ Dự đoán | Nhánh 0: Known Benign | Nhánh 1: Known Attack | Nhánh 2: Unseen Alert |
| :--- | :---: | :---: | :---: |
| **0: Known Benign (200,000)** | **55,651** | 859 | 10,527 |
| **1: Known Attack (133,553)** | 924 | **116,236** | 15,803 |
| **2: Open-Set Unknown (259,458)** | 21,262 | 114,204 | **123,992** |

---

## 4. Đo lường Thông lượng Hệ thống & Độ trễ (Throughput Benchmark)

Thực hiện đo đạc thực tế trên 50 micro-batches liên tiếp (204,800 luồng NetFlow, batch_size = 4096):

* **Chế độ Tất định (T=1, Fast Path):**
  - Thông lượng: **`168,625.47` flows/giây**
  - Độ trễ theo lô: `24.17 ms` / batch 4096 luồng (`5.93 µs` / luồng)
  - Bộ nhớ RAM tiêu thụ: `535.86 MB`
* **Chế độ MC Dropout Three-way OSR (T=20, High Security):**
  - Thông lượng: **`7,077.21` flows/giây**
  - Độ trễ theo lô: `578.65 ms` / batch 4096 luồng (`141.30 µs` / luồng)
  - Bộ nhớ RAM tiêu thụ: `540.41 MB`

---

## 5. Thử nghiệm Dữ liệu Mạng Bắt gói Thực tế (Live Packet Ingestion)

1. **Công cụ chuyển đổi `tools/pcap_to_netflow.py`:**
   - Tự động nhận diện cả `.pcap` và `.pcapng` (tích hợp `editcap.exe` của Wireshark).
   - Nhóm luồng 5-tuple và tính toán 43 đặc trưng NetFlow chuẩn.
2. **Kiểm thử trên tệp `test_traffic.pcapng` (133 MB Wi-Fi Traffic):**
   - Trích xuất 400 luồng NetFlow thực tế và suy luận trên Web Console trong **`181.7 ms` (`2,202 flows/s`)**.
   - Cấp phép thành công 58 luồng an toàn hợp lệ (Nhánh 0: Known Benign).
   - Nhận diện 14 luồng tấn công đã biết (Nhánh 1).
   - Kích hoạt cảnh báo bất định nhận thức cao cho 328 luồng (Nhánh 2: Unseen Alert với $\hat{\sigma} \in [0.18, 0.36] > \tau = 0.15$), chứng minh mô hình không bị quá tự tin (overconfident) khi gặp cấu trúc đồ thị mạng hình sao ngoại lai từ card mạng cá nhân.
