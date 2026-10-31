import gc
import os
import shutil
import sys
import time
import zipfile
import polars as pl

# Đảm bảo console trên Windows hiển thị đúng tiếng Việt (UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def ensure_raw_csv(base_dir: str = "data") -> str:
    """Kiểm tra file NF-UQ-NIDS-v2.csv, nếu chưa giải nén thì tự động giải nén từ file zip."""
    for candidate in [
        os.path.join(base_dir, "raw", "NF-UQ-NIDS-v2.csv"),
        os.path.join(base_dir, "NF-UQ-NIDS-v2.csv"),
    ]:
        if os.path.exists(candidate):
            return candidate
    target_csv = os.path.join(base_dir, "raw", "NF-UQ-NIDS-v2.csv")

    for root, _, files in os.walk("."):
        for f in files:
            if f.lower() == "nf-uq-nids-v2.csv":
                return os.path.join(root, f)

    zip_candidates = [
        os.path.join(base_dir, "raw", "NF-UQ-NIDS-v2.zip"),
        os.path.join(base_dir, "NF-UQ-NIDS-v2.zip"),
    ]
    zip_path = next((zp for zp in zip_candidates if os.path.exists(zp)), None)
    if zip_path is not None:
        print(f"[*] Tìm thấy '{zip_path}'. Đang giải nén NF-UQ-NIDS-v2.csv (~13.7 GB)...")
        os.makedirs(os.path.dirname(target_csv), exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            csv_members = [m for m in z.namelist() if m.endswith("NF-UQ-NIDS-v2.csv")]
            if not csv_members:
                raise FileNotFoundError("Không tìm thấy NF-UQ-NIDS-v2.csv trong file zip!")
            with z.open(csv_members[0]) as src, open(target_csv, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        print(f"[+] Đã giải nén xong: {target_csv}\n")
        return target_csv

    raise FileNotFoundError("Không tìm thấy NF-UQ-NIDS-v2.csv hoặc data/raw/NF-UQ-NIDS-v2.zip!")


def count_rows_and_cols(csv_path: str) -> tuple[int, int]:
    """Đếm nhanh số dòng và số cột của file CSV bằng Polars Streaming."""
    lf = pl.scan_csv(csv_path)
    n_cols = len(lf.collect_schema().names())
    n_rows = lf.select(pl.len()).collect(engine="streaming").item()
    return n_rows, n_cols


def main():
    start_time = time.time()
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    raw_csv = ensure_raw_csv(data_dir)

    buckets_dir = os.path.join(data_dir, "_tmp_buckets")
    seen_buckets_dir = os.path.join(data_dir, "_tmp_seen_buckets")

    test_unseen_csv = os.path.join(data_dir, "interim", "test_unseen.csv")
    train_csv = os.path.join(data_dir, "interim", "train.csv")
    val_csv = os.path.join(data_dir, "interim", "val.csv")
    test_seen_csv = os.path.join(data_dir, "interim", "test_seen.csv")
    os.makedirs(os.path.join(data_dir, "interim"), exist_ok=True)

    # Dọn dẹp thư mục tạm nếu còn sót từ lần chạy trước
    for d in (buckets_dir, seen_buckets_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(buckets_dir, exist_ok=True)
    os.makedirs(seen_buckets_dir, exist_ok=True)

    num_buckets = 64
    unseen_attacks = ["Bot", "Infilteration"]

    # =========================================================================
    # BƯỚC 1: Chia file CSV gốc (~13.7GB) thành 64 hash buckets theo struct hash
    #         Các dòng trùng lặp 100% sẽ luôn rơi vào cùng 1 bucket.
    # =========================================================================
    print(f"[1/4] Đang phân mảnh dữ liệu theo Hash ({num_buckets} buckets) bằng Streaming để tránh tràn RAM...")
    t0 = time.time()
    (
        pl.scan_csv(raw_csv)
        .with_columns(
            (pl.struct(pl.all()).hash(seed=0) % num_buckets)
            .cast(pl.UInt32)
            .alias("_bucket")
        )
        .sink_parquet(
            pl.PartitionBy(buckets_dir, key="_bucket", include_key=False),
            mkdir=True,
        )
    )
    print(f"      -> Hoàn tất phân mảnh ra đĩa trong {time.time() - t0:.1f}s.\n")
    gc.collect()

    # =========================================================================
    # BƯỚC 2: Xử lý từng bucket:
    #         - Loại bỏ hoàn toàn các dòng trùng lặp (unique trên toàn bộ cột)
    #         - Tách tập Unseen (Bot, Infilteration) -> ghi nối vào data/interim/test_unseen.csv
    #         - Phần Seen còn lại: Xáo trộn ngẫu nhiên (shuffle với seed=42)
    # =========================================================================
    print("[2/4] Đang lọc trùng lặp (drop duplicates), tách Unseen (Bot, Infilteration) và xáo trộn Seen (seed=42)...")
    t0 = time.time()

    total_dedup_rows = 0
    unseen_rows = 0
    seen_total_rows = 0
    n_cols = 0
    unseen_header_written = False

    # Duyệt cố định theo thứ tự bucket từ 0 đến num_buckets - 1 để đảm bảo tính tái lập (reproducibility)
    with open(test_unseen_csv, "wb") as f_unseen:
        for b_idx in range(num_buckets):
            b_folder = os.path.join(buckets_dir, f"_bucket={b_idx}")
            if not os.path.exists(b_folder):
                continue

            # Đọc toàn bộ file parquet trong bucket hiện tại (~150-200MB trên RAM) và xóa dòng trùng lặp
            df_bucket = pl.read_parquet(os.path.join(b_folder, "*.parquet")).unique(
                maintain_order=True
            )
            if n_cols == 0:
                n_cols = df_bucket.width

            total_dedup_rows += df_bucket.height

            # 1) Lọc Unseen: Bot hoặc Infilteration
            df_unseen = df_bucket.filter(pl.col("Attack").is_in(unseen_attacks))
            if df_unseen.height > 0:
                df_unseen.write_csv(f_unseen, include_header=not unseen_header_written)
                unseen_header_written = True
                unseen_rows += df_unseen.height
            del df_unseen

            # 2) Lọc Seen: Không chứa Bot và Infilteration, và Shuffle với seed=42
            df_seen = (
                df_bucket.filter(~pl.col("Attack").is_in(unseen_attacks))
                .sample(fraction=1.0, shuffle=True, seed=42)
            )
            if df_seen.height > 0:
                seen_part_path = os.path.join(seen_buckets_dir, f"seen_{b_idx:03d}.parquet")
                df_seen.write_parquet(seen_part_path)
                seen_total_rows += df_seen.height

            del df_seen
            del df_bucket
            # Xóa ngay thư mục bucket gốc sau khi xử lý xong để tiết kiệm ổ cứng
            shutil.rmtree(b_folder)
            gc.collect()

            if (b_idx + 1) % 16 == 0 or b_idx == num_buckets - 1:
                print(
                    f"      -> Đã xử lý {b_idx + 1}/{num_buckets} buckets | "
                    f"Unique: {total_dedup_rows:,} | Unseen: {unseen_rows:,} | Seen: {seen_total_rows:,}"
                )

    shutil.rmtree(buckets_dir, ignore_errors=True)
    print(f"      -> Hoàn tất Bước 2 trong {time.time() - t0:.1f}s.\n")

    # =========================================================================
    # BƯỚC 3: Chia tập Seen đã xáo trộn theo tỷ lệ 70% Train - 15% Val - 15% Test
    # =========================================================================
    print("[3/4] Đang chia tập Seen (70% Train, 15% Val, 15% Test) và ghi ra CSV...")
    t0 = time.time()

    train_target = int(seen_total_rows * 0.70)
    val_target = int(seen_total_rows * 0.15)
    test_seen_target = seen_total_rows - train_target - val_target

    print(f"      -> Mục tiêu số dòng Seen: Train = {train_target:,} | Val = {val_target:,} | Test_Seen = {test_seen_target:,}")

    # Vì các dòng đã được phân phối ngẫu nhiên vào 64 hash buckets và từng bucket cũng đã được shuffle(seed=42),
    # để đảm bảo xáo trộn toàn cục với seed=42, ta hoán vị thứ tự đọc các bucket bằng Polars sample(seed=42)
    seen_files = sorted(
        [
            os.path.join(seen_buckets_dir, f)
            for f in os.listdir(seen_buckets_dir)
            if f.endswith(".parquet")
        ]
    )
    shuffled_files = (
        pl.DataFrame({"path": seen_files})
        .sample(fraction=1.0, shuffle=True, seed=42)["path"]
        .to_list()
    )

    written_train = 0
    written_val = 0
    written_test_seen = 0

    with (
        open(train_csv, "wb") as f_train,
        open(val_csv, "wb") as f_val,
        open(test_seen_csv, "wb") as f_test,
    ):
        for p_file in shuffled_files:
            df_chunk = pl.read_parquet(p_file)
            offset = 0
            chunk_len = df_chunk.height

            # 1) Điền vào Train cho đủ train_target
            if written_train < train_target and offset < chunk_len:
                need = train_target - written_train
                take = min(need, chunk_len - offset)
                df_chunk.slice(offset, take).write_csv(
                    f_train, include_header=(written_train == 0)
                )
                written_train += take
                offset += take

            # 2) Điền vào Val cho đủ val_target
            if written_val < val_target and offset < chunk_len:
                need = val_target - written_val
                take = min(need, chunk_len - offset)
                df_chunk.slice(offset, take).write_csv(
                    f_val, include_header=(written_val == 0)
                )
                written_val += take
                offset += take

            # 3) Phần còn lại điền vào Test Seen
            if offset < chunk_len:
                take = chunk_len - offset
                df_chunk.slice(offset, take).write_csv(
                    f_test, include_header=(written_test_seen == 0)
                )
                written_test_seen += take
                offset += take

            del df_chunk
            os.remove(p_file)
            gc.collect()

    shutil.rmtree(seen_buckets_dir, ignore_errors=True)
    print(f"      -> Hoàn tất ghi 3 file Seen trong {time.time() - t0:.1f}s.\n")

    # =========================================================================
    # BƯỚC 4: Xác minh tổng số dòng (Shape) của 4 file đầu ra
    # =========================================================================
    print("[4/4] Đang kiểm tra lại số dòng thực tế (shape) của 4 file đầu ra...")
    n_train, c_train = count_rows_and_cols(train_csv)
    n_val, c_val = count_rows_and_cols(val_csv)
    n_test_seen, c_test_seen = count_rows_and_cols(test_seen_csv)
    n_test_unseen, c_test_unseen = count_rows_and_cols(test_unseen_csv)

    sum_outputs = n_train + n_val + n_test_seen + n_test_unseen

    print("=" * 72)
    print("KẾT QUẢ PHÂN CHIA DỮ LIỆU (INDUCTIVE SPLIT - UNSEEN ATTACK SIMULATION)")
    print("=" * 72)
    print(f"1. data/interim/train.csv       : {n_train:>12,} dòng | Shape: ({n_train}, {c_train})")
    print(f"2. data/interim/val.csv         : {n_val:>12,} dòng | Shape: ({n_val}, {c_val})")
    print(f"3. data/interim/test_seen.csv   : {n_test_seen:>12,} dòng | Shape: ({n_test_seen}, {c_test_seen})")
    print(f"4. data/interim/test_unseen.csv : {n_test_unseen:>12,} dòng | Shape: ({n_test_unseen}, {c_test_unseen})")
    print("-" * 72)
    print(f"Tổng số dòng sau lọc trùng (Unique) : {total_dedup_rows:>12,} dòng")
    print(f"Tổng số dòng của 4 file đầu ra      : {sum_outputs:>12,} dòng")
    print(f"Độ lệch (Hao hụt dữ liệu)           : {total_dedup_rows - sum_outputs:>12,} dòng")
    print(f"Tổng thời gian thực thi             : {time.time() - start_time:.1f} giây")
    print("=" * 72)


if __name__ == "__main__":
    main()
