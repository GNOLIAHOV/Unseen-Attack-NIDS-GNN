import os
import sys
import zipfile
import shutil
import polars as pl

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def find_or_extract_csv(base_dir: str = "data") -> str:
    """Scan for NF-UQ-NIDS-v2.csv, extracting it from NF-UQ-NIDS-v2.zip if needed."""
    for candidate in [
        os.path.join(base_dir, "raw", "NF-UQ-NIDS-v2.csv"),
        os.path.join(base_dir, "NF-UQ-NIDS-v2.csv"),
    ]:
        if os.path.exists(candidate):
            return candidate
    target_csv = os.path.join(base_dir, "raw", "NF-UQ-NIDS-v2.csv")

    # Search recursively in current workspace for any existing NF-UQ-NIDS-v2.csv
    for root, _, files in os.walk("."):
        for f in files:
            if f.lower() == "nf-uq-nids-v2.csv":
                return os.path.join(root, f)

    # Check if the zip archive exists in data/raw/ or data/ and extract the CSV
    zip_candidates = [
        os.path.join(base_dir, "raw", "NF-UQ-NIDS-v2.zip"),
        os.path.join(base_dir, "NF-UQ-NIDS-v2.zip"),
    ]
    zip_path = next((zp for zp in zip_candidates if os.path.exists(zp)), None)
    if zip_path is not None:
        print(f"[*] Tìm thấy file nén '{zip_path}'. Đang kiểm tra nội dung...")
        with zipfile.ZipFile(zip_path, "r") as z:
            csv_members = [
                m for m in z.namelist() if m.endswith("NF-UQ-NIDS-v2.csv")
            ]
            if not csv_members:
                raise FileNotFoundError(
                    "Không tìm thấy NF-UQ-NIDS-v2.csv bên trong file zip."
                )
            member = csv_members[0]
            print(
                f"[*] Đang giải nén '{member}' ra '{target_csv}' (~13.7 GB, vui lòng đợi)..."
            )
            os.makedirs(os.path.dirname(target_csv), exist_ok=True)
            with z.open(member) as src, open(target_csv, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            print(f"[+] Giải nén hoàn tất: {target_csv}\n")
            return target_csv

    raise FileNotFoundError(
        "Không tìm thấy file NF-UQ-NIDS-v2.csv hoặc data/raw/NF-UQ-NIDS-v2.zip"
    )


def main():
    pl.Config.set_tbl_rows(50)
    pl.Config.set_fmt_str_lengths(50)

    csv_path = find_or_extract_csv("data")
    print(f"[*] Đang phân tích file: {csv_path} bằng Polars (Lazy / Streaming)...\n")

    lf = pl.scan_csv(csv_path)

    # 1. Tổng số dòng, số IP nguồn & đích duy nhất
    print("[1/3] Đang tính tổng số bản ghi và số lượng IP Nguồn / Đích duy nhất...")
    summary_Stats = lf.select(
        pl.len().alias("total_rows"),
        pl.col("IPV4_SRC_ADDR").n_unique().alias("unique_src_ip"),
        pl.col("IPV4_DST_ADDR").n_unique().alias("unique_dst_ip"),
    ).collect(engine="streaming")

    total_rows = summary_Stats["total_rows"][0]
    unique_src = summary_Stats["unique_src_ip"][0]
    unique_dst = summary_Stats["unique_dst_ip"][0]

    print("=" * 60)
    print(f"Tổng số dòng (Flows)               : {total_rows:,}")
    print(f"Tổng số IP Nguồn duy nhất (SRC_IP) : {unique_src:,}")
    print(f"Tổng số IP Đích duy nhất (DST_IP)  : {unique_dst:,}")
    print("=" * 60)

    # 2. Thống kê phân bố nhãn (Label & Attack)
    print("\n[2/3] Đang thống kê phân bố nhãn tấn công (Label & Attack)...")
    label_dist = (
        lf.group_by("Label")
        .agg(pl.len().alias("Count"))
        .with_columns(
            (pl.col("Count") / total_rows * 100).round(4).alias("Percentage (%)")
        )
        .sort("Count", descending=True)
        .collect(engine="streaming")
    )
    print("\n--- Phân bố theo cột Label (0: Benign, 1: Attack) ---")
    print(label_dist)

    attack_dist = (
        lf.group_by(["Label", "Attack"])
        .agg(pl.len().alias("Count"))
        .with_columns(
            (pl.col("Count") / total_rows * 100).round(4).alias("Percentage (%)")
        )
        .sort("Count", descending=True)
        .collect(engine="streaming")
    )
    print("\n--- Phân bố chi tiết theo cột Attack ---")
    print(attack_dist)

    # 3. Đếm số dòng trùng lặp (sử dụng 64-bit struct hash kết hợp streaming để tối ưu RAM cho ~76 triệu dòng)
    print("\n[3/3] Đang đếm số dòng trùng lặp (Duplicate rows)...")
    unique_rows = (
        lf.select(pl.struct(pl.all()).hash(seed=0).alias("h"))
        .select(pl.col("h").n_unique())
        .collect(engine="streaming")
        .item()
    )
    duplicate_rows = total_rows - unique_rows
    dup_pct = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0.0

    print("=" * 60)
    print(f"Số dòng duy nhất (Unique rows)     : {unique_rows:,}")
    print(f"Số dòng trùng lặp (Duplicate rows) : {duplicate_rows:,} ({dup_pct:.4f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
