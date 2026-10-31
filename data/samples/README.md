# Data Samples (Tập Mẫu Dữ Liệu Kiểm Thử Thực Tế)

Thư mục này chứa các tệp mẫu lưu lượng mạng thực tế dùng để thử nghiệm nhanh hệ thống phát hiện xâm nhập NF-UQ-NIDS-v2 qua Web Console (`http://127.0.0.1:8000/`) hoặc công cụ chuyển đổi PCAP.

---

## 1. Danh sách Tệp Mẫu

| Tệp tin | Định dạng | Số luồng | Mục đích sử dụng |
| :--- | :--- | :--- | :--- |
| **`real_traffic_sample.csv`** | NetFlow CSV (chuẩn NF-UQ-NIDS-v2, 43 cột) | 500 luồng | Mẫu lưu lượng thực tế chưa qua tiền xử lý, chứa IP thực tế, cổng, cờ TCP và các chỉ số thống kê byte/packet. Dùng để kiểm thử khả năng tiền xử lý động (Dynamic Inference Preprocessing) của GATv2. |
| **`sample_external_wireshark_log.csv`** | CSV tối giản trích xuất từ Wireshark / NetFlow (8 cột) | 5 luồng | Minh họa cơ chế thích ứng trường dữ liệu (Fault-Tolerant Column Ingestion). Tự động ánh xạ bí danh (`src_ip`, `dst_ip`, `sport`, `dport`, `proto`, `bytes`, `pkts`, `duration`) và tự bù khuyết (auto-impute) 0.0 cho các trường còn thiếu. |
| **`my_traffic.pcapng`** *(tùy chọn cá nhân)* | PCAPNG packet capture | - | File bắt gói tin thực tế qua Wireshark từ card mạng cá nhân. Dùng làm đầu vào cho `tools/pcap_to_netflow.py`. |
| **`my_traffic.csv`** *(tùy chọn cá nhân)* | NetFlow CSV (43 cột) | 59 luồng | Kết quả chuyển đổi từ `my_traffic.pcapng` bằng `tools/pcap_to_netflow.py`. Sẵn sàng để tải trực tiếp lên Web Console. |

---

## 2. Hướng dẫn Thử nghiệm Nhanh

### Cách 1: Tải trực tiếp lên Web Console
1. Mở giao diện Web tại `http://127.0.0.1:8000/`.
2. Bấm nút **"Tải tệp .parquet / .csv"**.
3. Chọn tệp `data/samples/real_traffic_sample.csv` hoặc `data/samples/sample_external_wireshark_log.csv`.
4. Xem kết quả phân loại 3 nhánh, biểu đồ Epistemic Uncertainty và độ bất đồng Dual-Pathway.

### Cách 2: Chuyển đổi gói tin Wireshark mới sang NetFlow CSV
```bash
python tools/pcap_to_netflow.py data/samples/my_traffic.pcapng data/samples/my_traffic.csv
```
Sau đó tải tệp `data/samples/my_traffic.csv` vừa tạo lên Web Console.
