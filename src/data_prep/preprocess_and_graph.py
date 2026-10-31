import gc
import json
import os
import shutil
import sys
import time
import polars as pl

# Đảm bảo console trên Windows hiển thị đúng tiếng Việt (UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# 1. Phân nhóm các cột trong bộ dữ liệu NF-UQ-NIDS-v2
IP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
META_COLS = ["Label", "Attack", "Dataset"]

# Các cột phân loại (Categorical features)
CATEGORICAL_COLS = [
    "PROTOCOL",
    "L7_PROTO",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE",
    "FTP_COMMAND_RET_CODE",
]

# Các cột liên tục / số lượng (Numerical features)
NUMERICAL_COLS = [
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "DNS_QUERY_ID",
    "DNS_TTL_ANSWER",
]


def build_global_ip_mapping(csv_files: list[str], out_parquet: str) -> pl.DataFrame:
    """
    Quét toàn bộ IP Nguồn và IP Đích trên 4 tập dữ liệu bằng Polars Streaming
    để tạo bảng ánh xạ toàn cục: IP -> Node ID (0 .. N-1).
    """
    if os.path.exists(out_parquet):
        ip_df = pl.read_parquet(out_parquet)
        print(
            f"[1/4] Đã nạp từ điển Node ID toàn cục '{out_parquet}': "
            f"{ip_df.height:,} nodes (ID: 0 -> {ip_df.height - 1})\n"
        )
        return ip_df

    print("[1/4] Đang quét toàn bộ IP Nguồn & Đích để tạo từ điển Node ID toàn cục...")
    t0 = time.time()

    ip_lazy_list = []
    for path in csv_files:
        lf = pl.scan_csv(path)
        ip_lazy_list.append(lf.select(pl.col("IPV4_SRC_ADDR").alias("ip")))
        ip_lazy_list.append(lf.select(pl.col("IPV4_DST_ADDR").alias("ip")))

    ip_df = (
        pl.concat(ip_lazy_list)
        .unique()
        .drop_nulls()
        .sort("ip")
        .with_row_index("node_id")
        .with_columns(pl.col("node_id").cast(pl.Int32))
        .collect(engine="streaming")
    )

    ip_df.write_parquet(out_parquet)
    gc.collect()

    print(
        f"      -> Hoàn tất trong {time.time() - t0:.1f}s. "
        f"Tổng số Node (IP duy nhất toàn mạng): {ip_df.height:,}\n"
    )
    return ip_df


def fit_preprocessor_on_train(train_csv: str, artifacts_path: str) -> tuple[dict, dict]:
    """
    CHỐNG RÒ RỈ DỮ LIỆU (Data Leakage Prevention):
    Chỉ tính toán bộ từ vựng (vocab) cho các cột Categorical và (mean, std) cho các cột Numerical
    DỰA TRÊN DUY NHẤT tập data/interim/train.csv.
    """
    print("[2/4] Đang tính toán tham số tiền xử lý CHỈ TRÊN TẬP TRAIN (data/interim/train.csv)...")
    t0 = time.time()
    lf_train = pl.scan_csv(train_csv)

    # 1. Tính mean và std cho tất cả các cột Numerical trong 1 lần quét streaming duy nhất
    stat_exprs = []
    for c in NUMERICAL_COLS:
        stat_exprs.append(pl.col(c).cast(pl.Float64).mean().alias(f"{c}__mean"))
        stat_exprs.append(pl.col(c).cast(pl.Float64).std().alias(f"{c}__std"))

    print("      -> Đang tính Mean và Std cho 32 cột Numerical trên tập Train...")
    stats_row = lf_train.select(stat_exprs).collect(engine="streaming")

    num_stats = {}
    for c in NUMERICAL_COLS:
        mean_val = float(stats_row[f"{c}__mean"][0] or 0.0)
        std_val = float(stats_row[f"{c}__std"][0] or 1.0)
        if std_val == 0.0 or std_val != std_val:  # tránh chia cho 0 hoặc NaN
            std_val = 1.0
        num_stats[c] = {"mean": mean_val, "std": std_val}

    del stats_row
    gc.collect()

    # 2. Tính tập giá trị duy nhất (vocab) cho các cột Categorical trên tập Train
    print(f"      -> Đang trích xuất Vocabulary cho {len(CATEGORICAL_COLS)} cột Categorical trên tập Train...")
    cat_unique_exprs = [pl.col(c).unique().implode().alias(c) for c in CATEGORICAL_COLS]
    cat_uniques_row = lf_train.select(cat_unique_exprs).collect(engine="streaming")

    cat_vocabs = {}
    for c in CATEGORICAL_COLS:
        raw_vals = [v for v in cat_uniques_row[c][0].to_list() if v is not None]
        raw_vals.sort()
        # Ánh xạ các giá trị xuất hiện trong tập Train thành 1..K, dành 0 cho Unknown (OOV ở tập Val/Test)
        vocab_map = {val: idx + 1 for idx, val in enumerate(raw_vals)}
        cat_vocabs[c] = vocab_map
        print(f"         + {c:<22}: {len(vocab_map):>5} giá trị duy nhất (Train)")

    del cat_uniques_row
    gc.collect()

    # Lưu metadata tiền xử lý ra file JSON để phục vụ huấn luyện / kiểm tra sau này
    serializable_artifacts = {
        "numerical_stats": num_stats,
        "categorical_vocabs": {
            col: {str(k): v for k, v in vocab.items()}
            for col, vocab in cat_vocabs.items()
        },
    }
    with open(artifacts_path, "w", encoding="utf-8") as f:
        json.dump(serializable_artifacts, f, indent=2, ensure_ascii=False)

    print(f"      -> Hoàn tất Fit trên tập Train trong {time.time() - t0:.1f}s (Đã lưu '{artifacts_path}').\n")
    return cat_vocabs, num_stats


def transform_and_extract_edges(
    split_name: str,
    in_csv: str,
    out_parquet: str,
    src_map_df: pl.DataFrame,
    dst_map_df: pl.DataFrame,
    cat_vocabs: dict,
    num_stats: dict,
    chunk_size: int = 2_000_000,
) -> tuple[int, int]:
    """
    Xử lý theo từng chunk 2 triệu dòng bằng scan_csv().collect_batches() để đảm bảo RAM luôn thấp (<1.5GB),
    sau đó hợp nhất các chunk thành 1 file Parquet duy nhất bằng sink_parquet.
    """
    t0 = time.time()
    tmp_chunk_dir = f"{out_parquet}_chunks"
    if os.path.exists(tmp_chunk_dir):
        shutil.rmtree(tmp_chunk_dir)
    os.makedirs(tmp_chunk_dir, exist_ok=True)

    # Chuẩn bị danh sách biểu thức biến đổi đặc trưng
    transform_exprs = [
        pl.col("src_node_id").fill_null(-1).cast(pl.Int32),
        pl.col("dst_node_id").fill_null(-1).cast(pl.Int32),
        pl.col("Label").cast(pl.Int8).alias("label"),
        pl.col("Attack"),
        pl.col("Dataset"),
    ]

    for c in CATEGORICAL_COLS:
        vocab = cat_vocabs[c]
        transform_exprs.append(
            pl.col(c)
            .replace_strict(vocab, default=0, return_dtype=pl.Int32)
            .alias(c)
        )

    for c in NUMERICAL_COLS:
        mean_val = num_stats[c]["mean"]
        std_val = num_stats[c]["std"]
        transform_exprs.append(
            (
                (pl.col(c).cast(pl.Float64).fill_null(mean_val).fill_nan(mean_val) - mean_val)
                / std_val
            )
            .cast(pl.Float32)
            .alias(c)
        )

    # Duyệt từng batch từ scan_csv
    batches = pl.scan_csv(in_csv).collect_batches(chunk_size=chunk_size)
    total_rows = 0
    n_cols = 0

    for idx, df_batch in enumerate(batches):
        df_out = (
            df_batch.join(src_map_df, on="IPV4_SRC_ADDR", how="left")
            .join(dst_map_df, on="IPV4_DST_ADDR", how="left")
            .select(transform_exprs)
        )
        if n_cols == 0:
            n_cols = df_out.width
        total_rows += df_out.height

        part_file = os.path.join(tmp_chunk_dir, f"part_{idx:04d}.parquet")
        df_out.write_parquet(part_file)

        del df_batch
        del df_out
        gc.collect()

    # Gộp các part parquet thành 1 file parquet duy nhất bằng streaming sink_parquet
    if os.path.exists(out_parquet):
        os.remove(out_parquet)
    pl.scan_parquet(os.path.join(tmp_chunk_dir, "*.parquet")).sink_parquet(out_parquet)
    shutil.rmtree(tmp_chunk_dir, ignore_errors=True)
    gc.collect()

    print(
        f"      -> [{split_name:<9}] Đã tạo '{out_parquet}' | "
        f"Shape: ({total_rows:,}, {n_cols}) trong {time.time() - t0:.1f}s"
    )
    return total_rows, n_cols


def main():
    start_time = time.time()
    data_dir = "data"

    interim_dir = os.path.join(data_dir, "interim")
    proc_dir = os.path.join(data_dir, "processed")
    os.makedirs(proc_dir, exist_ok=True)

    splits = [
        ("Train", os.path.join(interim_dir, "train.csv"), os.path.join(proc_dir, "edges_train.parquet")),
        ("Val", os.path.join(interim_dir, "val.csv"), os.path.join(proc_dir, "edges_val.parquet")),
        ("Test Seen", os.path.join(interim_dir, "test_seen.csv"), os.path.join(proc_dir, "edges_test_seen.parquet")),
        ("Test Unseen", os.path.join(interim_dir, "test_unseen.csv"), os.path.join(proc_dir, "edges_test_unseen.parquet")),
    ]

    for _, in_csv, _ in splits:
        if not os.path.exists(in_csv):
            raise FileNotFoundError(f"Không tìm thấy file đầu vào: {in_csv}")

    ip_map_parquet = os.path.join(proc_dir, "ip_node_map.parquet")
    preproc_json = os.path.join(proc_dir, "preprocessor_params.json")

    # Bước 1: Tạo từ điển Node ID toàn cục cho mọi IP Nguồn & Đích
    ip_df = build_global_ip_mapping([s[1] for s in splits], ip_map_parquet)
    num_total_nodes = ip_df.height
    src_map_df = ip_df.rename({"ip": "IPV4_SRC_ADDR", "node_id": "src_node_id"})
    dst_map_df = ip_df.rename({"ip": "IPV4_DST_ADDR", "node_id": "dst_node_id"})
    del ip_df
    gc.collect()

    # Bước 2: Chỉ Fit bộ mã hóa Categorical & bộ chuẩn hóa Numerical trên tập Train
    cat_vocabs, num_stats = fit_preprocessor_on_train(splits[0][1], preproc_json)

    # Bước 3: Biến đổi (Transform) cả 4 tập và trích xuất danh sách cạnh (Edges) cho PyTorch Geometric
    print("[3/4] Đang biến đổi (Transform) theo Chunking và trích xuất danh sách cạnh (Edges)...")
    results = []
    for name, in_csv, out_parquet in splits:
        rows, cols = transform_and_extract_edges(
            name, in_csv, out_parquet, src_map_df, dst_map_df, cat_vocabs, num_stats
        )
        results.append((name, out_parquet, rows, cols))

    del src_map_df
    del dst_map_df
    gc.collect()

    # Bước 4: In báo cáo tổng hợp
    print("\n[4/4] Tổng kết dữ liệu đồ thị (PyTorch Geometric Ready):")
    print("=" * 80)
    print("BÁO CÁO TIỀN XỬ LÝ & TRÍCH XUẤT ĐỒ THỊ (PREPROCESS & GRAPH EXTRACTION)")
    print("=" * 80)
    print(f"Tổng số Node (IP duy nhất trên toàn mạng) : {num_total_nodes:,} nodes (ID: 0 -> {num_total_nodes - 1})")
    print(f"Số cột đặc trưng cạnh (Edge Features)     : {len(CATEGORICAL_COLS) + len(NUMERICAL_COLS)} cột ({len(CATEGORICAL_COLS)} Categorical + {len(NUMERICAL_COLS)} Numerical)")
    print(f"Cấu trúc cột trong mỗi file cạnh (46 cột) : src_node_id, dst_node_id, label, Attack, Dataset + 41 features")
    print("-" * 80)
    total_edges = 0
    for name, out_parquet, rows, cols in results:
        total_edges += rows
        print(f"- {name:<12} ({out_parquet:<29}) : {rows:>12,} cạnh | Shape: ({rows}, {cols})")
    print("-" * 80)
    print(f"Tổng số cạnh (Edges) trên toàn bộ 4 tập   : {total_edges:>12,} cạnh")
    print(f"Tổng thời gian thực thi                   : {time.time() - start_time:.1f} giây")
    print("=" * 80)


if __name__ == "__main__":
    main()
