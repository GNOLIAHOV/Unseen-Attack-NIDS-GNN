import gc
import json
import os
import sys
import time
import warnings
import numpy as np
import polars as pl
import psutil
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.gnn_architectures import (
    EDGE_ATTR_DIM,
    HIDDEN_DIM,
    NODE_INIT_DIM,
    NON_FEATURE_COLS,
    NUM_HEADS,
    NUM_TOTAL_NODES,
    GNNInductiveGATv2,
    compute_inductive_node_features,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Đảm bảo console trên Windows hiển thị đúng tiếng Việt (UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


BATCH_SIZE = 4096
NUM_NEIGHBORS = [10, 10]
WARMUP_BATCHES = 5
BENCHMARK_BATCHES = 50
MC_SAMPLES_T = 20


def load_graph_split(
    parquet_path: str,
    max_edges: int = 250_000,
) -> Data:
    """Đọc tập mẫu từ data/processed/edges_test_seen.parquet đủ cho Warm-up + 50 batches."""
    lf = pl.scan_parquet(parquet_path).head(max_edges)
    schema_cols = lf.collect_schema().names()
    feature_cols = [c for c in schema_cols if c not in NON_FEATURE_COLS]

    df_topo = lf.select(["src_node_id", "dst_node_id", "label"]).collect(engine="streaming")
    src_np = df_topo["src_node_id"].to_numpy().astype(np.int64, copy=False)
    dst_np = df_topo["dst_node_id"].to_numpy().astype(np.int64, copy=False)
    y_np = df_topo["label"].to_numpy().astype(np.float32, copy=False)
    del df_topo
    gc.collect()

    edge_index = torch.from_numpy(np.vstack([src_np, dst_np]))
    del src_np, dst_np
    y = torch.from_numpy(y_np)
    del y_np
    gc.collect()

    df_feat = (
        lf.select([pl.col(c).cast(pl.Float32) for c in feature_cols])
        .collect(engine="streaming")
    )
    edge_attr = torch.from_numpy(df_feat.to_numpy())
    del df_feat
    gc.collect()

    node_x = compute_inductive_node_features(edge_index, num_nodes=NUM_TOTAL_NODES)
    return Data(
        x=node_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )


def get_process_rss_mb() -> float:
    """Lấy lượng bộ nhớ vật lý (RSS RAM) hiện tại của tiến trình tính bằng MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


@torch.no_grad()
def run_benchmark_scenario(
    model: nn.Module,
    loader: LinkNeighborLoader,
    full_target_edge_attr: torch.Tensor,
    device: torch.device,
    mc_passes: int = 1,
    warmup_batches: int = WARMUP_BATCHES,
    measure_batches: int = BENCHMARK_BATCHES,
) -> dict:
    if mc_passes == 1:
        model.eval()
    else:
        model.train()

    iterator = iter(loader)

    for _ in range(warmup_batches):
        batch = next(iterator).to(device)
        target_ea = full_target_edge_attr[batch.input_id.cpu()].to(device)
        for _ in range(mc_passes):
            _ = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_label_index=batch.edge_label_index,
                target_edge_attr=target_ea,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    peak_ram_mb = get_process_rss_mb()
    batch_latencies_ms = []
    total_flows = 0

    t_start = time.perf_counter()
    for _ in range(measure_batches):
        b_start = time.perf_counter()

        batch = next(iterator).to(device)
        target_ea = full_target_edge_attr[batch.input_id.cpu()].to(device)
        batch_flows = batch.edge_label_index.size(1)

        if mc_passes == 1:
            _ = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_label_index=batch.edge_label_index,
                target_edge_attr=target_ea,
            )
        else:
            h, bi_ei, bi_ea, eli, e_target = model.encode_subgraph(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_label_index=batch.edge_label_index,
                target_edge_attr=target_ea,
            )
            mc_preds = []
            for _ in range(mc_passes):
                prob_t = model.forward_single_mc(h, bi_ei, bi_ea, eli, e_target)
                mc_preds.append(prob_t)
            mc_stack = torch.stack(mc_preds, dim=0)
            _ = mc_stack.mean(dim=0)
            _ = mc_stack.std(dim=0, unbiased=False)

        if device.type == "cuda":
            torch.cuda.synchronize()

        b_end = time.perf_counter()
        batch_latencies_ms.append((b_end - b_start) * 1000.0)
        total_flows += int(batch_flows)

        cur_ram = get_process_rss_mb()
        if cur_ram > peak_ram_mb:
            peak_ram_mb = cur_ram

    total_time_sec = time.perf_counter() - t_start
    avg_latency_ms = float(np.mean(batch_latencies_ms))
    p95_latency_ms = float(np.percentile(batch_latencies_ms, 95))
    throughput_fps = total_flows / total_time_sec

    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    return {
        "mc_passes_T": mc_passes,
        "measured_batches": measure_batches,
        "total_flows_processed": total_flows,
        "total_time_sec": round(total_time_sec, 4),
        "avg_batch_latency_ms": round(avg_latency_ms, 2),
        "p95_batch_latency_ms": round(p95_latency_ms, 2),
        "per_flow_latency_us": round((total_time_sec * 1e6) / total_flows, 2),
        "throughput_flows_per_sec": round(throughput_fps, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "peak_ram_mb": round(peak_ram_mb, 2),
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join("models", "gnns", "gatv2_g1_v2_model.pt")
    test_seen_path = os.path.join("data", "processed", "edges_test_seen.parquet")
    out_json_path = os.path.join("results", "metrics", "throughput_benchmark.json")

    required_edges = (WARMUP_BATCHES + BENCHMARK_BATCHES) * BATCH_SIZE + 20_000

    print("=" * 94)
    print("ĐO LƯỜNG HIỆU NĂNG HỆ THỐNG (SYSTEM THROUGHPUT & LATENCY BENCHMARK - TUẦN 13)")
    print("=" * 94)
    print(f"- Thiết bị thực thi (Device)   : {device.type.upper()}")
    print(f"- Mô hình đánh giá             : {model_path} (GATv2Conv + LayerNorm)")
    print(f"- Cấu hình LinkNeighborLoader  : batch_size={BATCH_SIZE}, num_neighbors={NUM_NEIGHBORS}")
    print(f"- Giao thức Benchmark          : {WARMUP_BATCHES} Warm-up batches + {BENCHMARK_BATCHES} Measured batches ({BENCHMARK_BATCHES * BATCH_SIZE:,} luồng)\n")

    print("[1/3] Đang nạp mô hình GATv2 và dữ liệu đồ thị từ 'data/processed/edges_test_seen.parquet'...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = GNNInductiveGATv2(
        node_init_dim=checkpoint.get("node_init_dim", NODE_INIT_DIM),
        hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
        heads=checkpoint.get("heads", NUM_HEADS),
        edge_attr_dim=checkpoint.get("edge_attr_dim", EDGE_ATTR_DIM),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    graph_data = load_graph_split(test_seen_path, max_edges=required_edges)

    loader = LinkNeighborLoader(
        data=graph_data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        edge_label_index=graph_data.edge_index,
        edge_label=graph_data.y,
        shuffle=False,
    )
    print(f"      -> Đã chuẩn bị {graph_data.edge_index.size(1):,} cạnh ({len(loader)} batches).\n")

    print(f"[2/3] Đang đo Kịch bản 1: Standard Inference (T=1, model.eval(), {WARMUP_BATCHES} warmup + {BENCHMARK_BATCHES} batches)...")
    res_t1 = run_benchmark_scenario(
        model,
        loader,
        graph_data.edge_attr,
        device,
        mc_passes=1,
        warmup_batches=WARMUP_BATCHES,
        measure_batches=BENCHMARK_BATCHES,
    )
    print(
        f"      -> Hoàn tất T=1: Tổng thời gian = {res_t1['total_time_sec']:.2f}s | "
        f"Latency = {res_t1['avg_batch_latency_ms']:.2f} ms/batch | "
        f"Throughput = {res_t1['throughput_flows_per_sec']:,.1f} flows/s\n"
    )

    print(f"[3/3] Đang đo Kịch bản 2: MC Dropout Inference (T={MC_SAMPLES_T}, {WARMUP_BATCHES} warmup + {BENCHMARK_BATCHES} batches)...")
    res_t20 = run_benchmark_scenario(
        model,
        loader,
        graph_data.edge_attr,
        device,
        mc_passes=MC_SAMPLES_T,
        warmup_batches=WARMUP_BATCHES,
        measure_batches=BENCHMARK_BATCHES,
    )
    print(
        f"      -> Hoàn tất T={MC_SAMPLES_T}: Tổng thời gian = {res_t20['total_time_sec']:.2f}s | "
        f"Latency = {res_t20['avg_batch_latency_ms']:.2f} ms/batch | "
        f"Throughput = {res_t20['throughput_flows_per_sec']:,.1f} flows/s\n"
    )

    slowdown_factor = res_t1["throughput_flows_per_sec"] / res_t20["throughput_flows_per_sec"]
    latency_ratio = res_t20["avg_batch_latency_ms"] / res_t1["avg_batch_latency_ms"]

    summary = {
        "device": device.type,
        "batch_size": BATCH_SIZE,
        "num_neighbors": NUM_NEIGHBORS,
        "warmup_batches": WARMUP_BATCHES,
        "measured_batches": BENCHMARK_BATCHES,
        "scenario_1_standard_T1": res_t1,
        "scenario_2_mc_dropout_T20": res_t20,
        "tradeoff_analysis": {
            "throughput_slowdown_factor": round(slowdown_factor, 2),
            "latency_increase_factor": round(latency_ratio, 2),
        },
    }

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    mem_label = "Peak VRAM (MB)" if device.type == "cuda" else "Peak RAM (MB)"
    mem_t1 = res_t1["peak_vram_mb"] if device.type == "cuda" else res_t1["peak_ram_mb"]
    mem_t20 = res_t20["peak_vram_mb"] if device.type == "cuda" else res_t20["peak_ram_mb"]

    print("=" * 94)
    print("BẢNG SO SÁNH HIỆU NĂNG HỆ THỐNG: STANDARD INFERENCE (T=1) vs MC DROPOUT UQ (T=20)")
    print("=" * 94)
    print(
        f"{'Kịch bản đo (50 Batches = 204,800 Flows)':<40} | {'Throughput (Flows/s)':>20} | {'Batch Latency (ms)':>18} | {mem_label:>14}"
    )
    print("-" * 94)
    print(
        f"{'1. Standard Inference (T=1, Deterministic)':<40} | "
        f"{res_t1['throughput_flows_per_sec']:>20,.2f} | "
        f"{res_t1['avg_batch_latency_ms']:>15.2f} ms | "
        f"{mem_t1:>11.2f} MB"
    )
    print(
        f"{'2. MC Dropout UQ (T=20, Three-way Ready)':<40} | "
        f"{res_t20['throughput_flows_per_sec']:>20,.2f} | "
        f"{res_t20['avg_batch_latency_ms']:>15.2f} ms | "
        f"{mem_t20:>11.2f} MB"
    )
    print("-" * 94)
    print("=" * 94)


if __name__ == "__main__":
    main()
