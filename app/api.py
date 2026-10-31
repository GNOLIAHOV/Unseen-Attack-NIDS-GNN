from contextlib import asynccontextmanager
import io
import json
import os
import sys
import time
from typing import Any
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
import polars as pl
from pydantic import BaseModel, Field
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.gnn_architectures import (
    EDGE_ATTR_DIM,
    HIDDEN_DIM,
    NODE_INIT_DIM,
    NUM_HEADS,
    NUM_TOTAL_NODES,
    GNNInductiveGATv2,
    compute_inductive_node_features,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 1. CẤU HÌNH HỆ THỐNG & ĐẶC TRƯNG NETFLOW (41 FEATURES)
# =============================================================================
MODEL_PATH = os.path.join("models", "gnns", "gatv2_g1_v2_model.pt")
PREPROC_PATH = os.path.join("data", "processed", "preprocessor_params.json")
DEMO_PARQUET_PATH = os.path.join("data", "processed", "demo_microbatch.parquet")
METRICS_DIR = os.path.join("results", "metrics")

DEFAULT_MC_SAMPLES_T = 20
DEFAULT_STD_THRESHOLD = 0.15

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

FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

LABEL_NAMES = {
    0: "Benign (Trusted)",
    1: "Known Attack",
    2: "Unseen Attack Alert (High Epistemic Uncertainty)",
}


# =============================================================================
# 2. QUẢN LÝ TRẠNG THÁI MODEL TRONG BỘ NHỚ (LIFESPAN / STARTUP)
# =============================================================================
class NIDSRuntimeEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: GNNInductiveGATv2 | None = None
        self.preproc_params: dict[str, Any] | None = None
        self.ip_to_node_id: dict[str, int] = {}
        self.bg_edge_index: torch.Tensor | None = None
        self.bg_edge_attr: torch.Tensor | None = None
        self.ref_node_x: torch.Tensor | None = None
        self.bg_num_nodes: int = NUM_TOTAL_NODES
        self.loaded_at: float = 0.0

    def load_resources(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Không tìm thấy checkpoint mô hình tại '{MODEL_PATH}'")

        checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        model = GNNInductiveGATv2(
            node_init_dim=checkpoint.get("node_init_dim", NODE_INIT_DIM),
            hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
            heads=checkpoint.get("heads", NUM_HEADS),
            edge_attr_dim=checkpoint.get("edge_attr_dim", EDGE_ATTR_DIM),
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model

        if os.path.exists(PREPROC_PATH):
            with open(PREPROC_PATH, "r", encoding="utf-8") as f:
                self.preproc_params = json.load(f)
            cat_vocabs = self.preproc_params.get("categorical_vocabs", {})
            for col_name, vocab_dict in cat_vocabs.items():
                expanded = {}
                for k, v in vocab_dict.items():
                    expanded[str(k)] = v
                    try:
                        fk = float(k)
                        if fk.is_integer():
                            expanded[str(int(fk))] = v
                            expanded[f"{int(fk)}.0"] = v
                    except (ValueError, TypeError):
                        pass
                cat_vocabs[col_name] = expanded

        ip_map_path = os.path.join("data", "processed", "ip_node_map.parquet")
        if os.path.exists(ip_map_path):
            df_ip = pl.read_parquet(ip_map_path)
            if "ip" in df_ip.columns and "node_id" in df_ip.columns:
                self.ip_to_node_id = dict(
                    zip(df_ip["ip"].to_list(), df_ip["node_id"].to_list())
                )

        bg_parquet = os.path.join("data", "processed", "edges_test_seen.parquet")
        tu_parquet = os.path.join("data", "processed", "edges_test_unseen.parquet")
        if os.path.exists(bg_parquet):
            df_bg = pl.scan_parquet(bg_parquet).head(200_000).collect(engine="streaming")
            s_np = df_bg["src_node_id"].to_numpy().astype(np.int64, copy=False)
            d_np = df_bg["dst_node_id"].to_numpy().astype(np.int64, copy=False)
            ei_seen = torch.from_numpy(np.vstack([s_np, d_np])).to(self.device)
            x_seen = compute_inductive_node_features(ei_seen, num_nodes=self.bg_num_nodes)

            if os.path.exists(tu_parquet):
                df_tu = (
                    pl.scan_parquet(tu_parquet)
                    .head(200_000)
                    .collect(engine="streaming")
                )
                ei_tu = torch.from_numpy(
                    np.vstack(
                        [
                            df_tu["src_node_id"].to_numpy().astype(np.int64, copy=False),
                            df_tu["dst_node_id"].to_numpy().astype(np.int64, copy=False),
                        ]
                    )
                ).to(self.device)
                x_tu = compute_inductive_node_features(ei_tu, num_nodes=self.bg_num_nodes)
                self.ref_node_x = torch.maximum(x_seen, x_tu)

                ea_seen_bg = torch.from_numpy(
                    df_bg.head(30_000)
                    .select([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS])
                    .to_numpy()
                ).to(self.device)
                ea_tu_bg = torch.from_numpy(
                    df_tu.head(30_000)
                    .select([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS])
                    .to_numpy()
                ).to(self.device)
                self.bg_edge_index = torch.cat([ei_seen[:, :30_000], ei_tu[:, :30_000]], dim=1)
                self.bg_edge_attr = torch.cat([ea_seen_bg, ea_tu_bg], dim=0)
                del df_tu, ei_tu, x_tu, ea_seen_bg, ea_tu_bg
            else:
                self.ref_node_x = x_seen
                self.bg_edge_index = ei_seen[:, :50_000]
                self.bg_edge_attr = (
                    torch.from_numpy(
                        df_bg.head(50_000)
                        .select([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS])
                        .to_numpy()
                    ).to(self.device)
                )
            del df_bg, ei_seen, x_seen

        self.loaded_at = time.time()

    @torch.no_grad()
    def run_three_way_inference(
        self,
        src_nodes: list[Any],
        dst_nodes: list[Any],
        edge_features_np: np.ndarray,
        mc_samples: int = DEFAULT_MC_SAMPLES_T,
        std_threshold: float = DEFAULT_STD_THRESHOLD,
    ) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Mô hình chưa được khởi tạo trong bộ nhớ!")

        num_flows = len(src_nodes)
        if num_flows == 0:
            return []

        parsed_src = []
        parsed_dst = []
        for s, d in zip(src_nodes, dst_nodes):
            try:
                s_id = int(s) % self.bg_num_nodes
            except (ValueError, TypeError):
                s_id = abs(hash(str(s))) % self.bg_num_nodes
            try:
                d_id = int(d) % self.bg_num_nodes
            except (ValueError, TypeError):
                d_id = abs(hash(str(d))) % self.bg_num_nodes
            parsed_src.append(s_id)
            parsed_dst.append(d_id)

        batch_ei = torch.tensor([parsed_src, parsed_dst], dtype=torch.long, device=self.device)
        batch_ea = torch.from_numpy(edge_features_np.astype(np.float32, copy=False)).to(self.device)

        if self.bg_edge_index is not None and self.bg_edge_attr is not None:
            seed_nodes = torch.unique(batch_ei)
            mask = torch.isin(self.bg_edge_index[0], seed_nodes) | torch.isin(
                self.bg_edge_index[1], seed_nodes
            )
            nbr_ei = self.bg_edge_index[:, mask][:, :2000]
            nbr_ea = self.bg_edge_attr[mask][:2000]
            full_ei = torch.cat([batch_ei, nbr_ei], dim=1)
            full_ea = torch.cat([batch_ea, nbr_ea], dim=0)
        else:
            full_ei = batch_ei
            full_ea = batch_ea

        unique_nodes, relabeled = torch.unique(full_ei, return_inverse=True)
        local_full_ei = relabeled.view(2, -1)
        local_target_ei = local_full_ei[:, :num_flows]

        x_batch = compute_inductive_node_features(batch_ei, num_nodes=self.bg_num_nodes)
        if self.ref_node_x is not None:
            x_global = torch.maximum(self.ref_node_x, x_batch)
        else:
            x_global = x_batch
        x = x_global[unique_nodes]

        self.model.eval()
        h, bi_ei, bi_ea, eli, target_e_norm = self.model.encode_subgraph(
            x=x,
            edge_index=local_full_ei,
            edge_attr=full_ea,
            edge_label_index=local_target_ei,
            target_edge_attr=batch_ea,
        )
        l_topo, l_flow = self.model.forward_dual_logits(h, bi_ei, bi_ea, eli, target_e_norm)
        p_topo = torch.sigmoid(l_topo).view(-1).cpu().numpy()
        p_flow = torch.sigmoid(l_flow).view(-1).cpu().numpy()
        discrepancy = np.abs(p_topo - p_flow)

        self.model.train()
        mc_preds = []
        for _ in range(mc_samples):
            prob_t = self.model.forward_single_mc(h, bi_ei, bi_ea, eli, target_e_norm)
            mc_preds.append(prob_t.view(-1))

        mc_stack = torch.stack(mc_preds, dim=0)
        mean_probs = mc_stack.mean(dim=0).cpu().numpy()
        epistemic_stds = mc_stack.std(dim=0, unbiased=False).cpu().numpy()
        osr_scores = np.maximum(epistemic_stds, 0.35 * discrepancy)

        predictions = []
        for idx in range(num_flows):
            m_prob = float(mean_probs[idx])
            e_std = float(epistemic_stds[idx])
            p_t = float(p_topo[idx])
            p_f = float(p_flow[idx])
            disc = float(discrepancy[idx])
            osr_s = float(osr_scores[idx])

            if e_std > std_threshold or osr_s > std_threshold:
                pred_code = 2
            elif m_prob >= 0.5:
                pred_code = 1
            else:
                pred_code = 0

            predictions.append(
                {
                    "flow_index": idx,
                    "src_node": str(src_nodes[idx]),
                    "dst_node": str(dst_nodes[idx]),
                    "predicted_label": pred_code,
                    "predicted_class": LABEL_NAMES[pred_code],
                    "action": "BLOCK / ALERT" if pred_code in (1, 2) else "ALLOW",
                    "mean_prob": round(m_prob, 6),
                    "epistemic_std": round(e_std, 6),
                    "topo_prob": round(p_t, 6),
                    "flow_prob": round(p_f, 6),
                    "pathway_discrepancy": round(disc, 6),
                    "osr_score": round(osr_s, 6),
                }
            )
        return predictions


engine = NIDSRuntimeEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo và nạp sẵn mô hình GATv2 vào RAM/VRAM ngay khi FastAPI khởi động."""
    t0 = time.time()
    engine.load_resources()
    print(
        f"[+] Đã nạp sẵn mô hình '{MODEL_PATH}' trên {engine.device} trong {time.time() - t0:.2f}s."
    )
    yield


app = FastAPI(
    title="NF-UQ-NIDS-v2 Open-Set & Unseen Attack Detection API",
    description=(
        "Backend API phát hiện xâm nhập mạng và Nhận dạng Tập Mở (Open-Set Recognition) "
        "sử dụng kiến trúc Inductive Edge-Aware GATv2 kết hợp Định lượng bất định 3 nhánh (Three-way MC Dropout UQ)."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FlowItem(BaseModel):
    src_node: str | int = Field(default="0", description="Địa chỉ IP nguồn hoặc src_node_id")
    dst_node: str | int = Field(default="1", description="Địa chỉ IP đích hoặc dst_node_id")
    features: list[float] | dict[str, float] = Field(
        ...,
        description="Danh sách 41 đặc trưng NetFlow đã chuẩn hóa (list 41 số thực) hoặc dict tên cột -> giá trị",
    )


class PredictBatchRequest(BaseModel):
    flows: list[FlowItem] = Field(..., description="Danh sách các luồng NetFlow trong micro-batch")
    mc_samples: int = Field(
        default=DEFAULT_MC_SAMPLES_T, ge=2, le=100, description="Số lần lấy mẫu MC Dropout (T)"
    )
    std_threshold: float = Field(
        default=DEFAULT_STD_THRESHOLD,
        gt=0.0,
        lt=1.0,
        description="Ngưỡng Epistemic Std / OSR phát hiện Unseen Attack",
    )


COMMON_ALIASES = {
    "ipv4_src_addr": "IPV4_SRC_ADDR", "src_ip": "IPV4_SRC_ADDR", "source_ip": "IPV4_SRC_ADDR", "ip.src": "IPV4_SRC_ADDR", "src_addr": "IPV4_SRC_ADDR",
    "ipv4_dst_addr": "IPV4_DST_ADDR", "dst_ip": "IPV4_DST_ADDR", "dest_ip": "IPV4_DST_ADDR", "destination_ip": "IPV4_DST_ADDR", "ip.dst": "IPV4_DST_ADDR", "dst_addr": "IPV4_DST_ADDR",
    "l4_src_port": "L4_SRC_PORT", "src_port": "L4_SRC_PORT", "sport": "L4_SRC_PORT", "source_port": "L4_SRC_PORT", "tcp.srcport": "L4_SRC_PORT", "udp.srcport": "L4_SRC_PORT",
    "l4_dst_port": "L4_DST_PORT", "dst_port": "L4_DST_PORT", "dport": "L4_DST_PORT", "dest_port": "L4_DST_PORT", "tcp.dstport": "L4_DST_PORT", "udp.dstport": "L4_DST_PORT",
    "protocol": "PROTOCOL", "proto": "PROTOCOL", "ip.proto": "PROTOCOL",
    "in_bytes": "IN_BYTES", "bytes": "IN_BYTES", "tot_bytes": "IN_BYTES", "src_bytes": "IN_BYTES", "total_bytes": "IN_BYTES",
    "in_pkts": "IN_PKTS", "pkts": "IN_PKTS", "packets": "IN_PKTS", "tot_pkts": "IN_PKTS", "src_pkts": "IN_PKTS", "total_packets": "IN_PKTS",
    "out_bytes": "OUT_BYTES", "dst_bytes": "OUT_BYTES",
    "out_pkts": "OUT_PKTS", "dst_pkts": "OUT_PKTS",
    "flow_duration_milliseconds": "FLOW_DURATION_MILLISECONDS", "duration": "FLOW_DURATION_MILLISECONDS", "flow_duration": "FLOW_DURATION_MILLISECONDS",
    "tcp_flags": "TCP_FLAGS", "flags": "TCP_FLAGS",
    "min_ttl": "MIN_TTL", "ttl": "MIN_TTL", "ip.ttl": "MIN_TTL",
    "max_ttl": "MAX_TTL",
}


def _format_dataframe_response(
    df: pl.DataFrame,
    filename: str,
    mc_samples: int,
    std_threshold: float,
    t0: float,
) -> dict[str, Any]:
    # 1. Tự động tương thích tên cột: nhận diện các biến thể tên cột thực tế (Wireshark, Zeek, nfdump)
    rename_map = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for alias_low, target_col in COMMON_ALIASES.items():
        if target_col not in df.columns and alias_low in lower_cols:
            rename_map[lower_cols[alias_low]] = target_col
    if rename_map:
        df = df.rename(rename_map)

    # 2. Bù đắp đặc trưng mở rộng nếu tải lên file log mạng bên ngoài
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    has_network_fields = any(
        c in df.columns
        for c in [
            "L4_SRC_PORT",
            "L4_DST_PORT",
            "PROTOCOL",
            "IN_BYTES",
            "IN_PKTS",
            "IPV4_SRC_ADDR",
            "src_node_id",
        ]
    )
    if missing_cols:
        if has_network_fields:
            add_exprs = [pl.lit(0.0).alias(c) for c in missing_cols]
            df = df.with_columns(add_exprs)
        else:
            raise HTTPException(
                status_code=422,
                detail=f"File không chứa các đặc trưng mạng NetFlow hợp lệ (thiếu: {missing_cols[:5]}...).",
            )

    if "src_node_id" in df.columns and "dst_node_id" in df.columns:
        src_nodes = df["src_node_id"].to_list()
        dst_nodes = df["dst_node_id"].to_list()
    elif "IPV4_SRC_ADDR" in df.columns and "IPV4_DST_ADDR" in df.columns:
        raw_src = df["IPV4_SRC_ADDR"].to_list()
        raw_dst = df["IPV4_DST_ADDR"].to_list()
        src_nodes = [
            engine.ip_to_node_id.get(str(ip), abs(hash(str(ip))) % engine.bg_num_nodes)
            for ip in raw_src
        ]
        dst_nodes = [
            engine.ip_to_node_id.get(str(ip), abs(hash(str(ip))) % engine.bg_num_nodes)
            for ip in raw_dst
        ]
    else:
        src_nodes = list(range(df.height))
        dst_nodes = [i + 1 for i in range(df.height)]

    # Tự động phát hiện và chuẩn hóa nếu file tải lên là dữ liệu thô chưa qua Z-score (ví dụ từ data/interim/)
    is_raw_unnormalized = "IPV4_SRC_ADDR" in df.columns or (
        float(df["L4_SRC_PORT"].cast(pl.Float64).abs().max() or 0.0) > 50.0
    )
    if is_raw_unnormalized and engine.preproc_params is not None:
        cat_vocabs = engine.preproc_params.get("categorical_vocabs", {})
        num_stats = engine.preproc_params.get("numerical_stats", {})
        exprs = []
        for c in CATEGORICAL_COLS:
            vocab_str_map = cat_vocabs.get(c, {})
            exprs.append(
                pl.col(c)
                .cast(pl.Utf8)
                .replace_strict(vocab_str_map, default=0, return_dtype=pl.Int32)
                .cast(pl.Float32)
                .alias(c)
            )
        for c in NUMERICAL_COLS:
            st_info = num_stats.get(c, {"mean": 0.0, "std": 1.0})
            m_val = float(st_info.get("mean", 0.0))
            s_val = float(st_info.get("std", 1.0)) or 1.0
            exprs.append(
                (
                    (pl.col(c).cast(pl.Float64).fill_null(m_val).fill_nan(m_val) - m_val)
                    / s_val
                )
                .cast(pl.Float32)
                .alias(c)
            )
        edge_features_np = df.select(exprs).to_numpy()
    else:
        edge_features_np = df.select([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS]).to_numpy()

    results = engine.run_three_way_inference(
        src_nodes=src_nodes,
        dst_nodes=dst_nodes,
        edge_features_np=edge_features_np,
        mc_samples=mc_samples,
        std_threshold=std_threshold,
    )

    if "label" in df.columns:
        gt_labels = df["label"].to_list()
        for r, gt in zip(results, gt_labels):
            r["ground_truth_label"] = int(gt)
    elif "Label" in df.columns:
        gt_labels = df["Label"].to_list()
        for r, gt in zip(results, gt_labels):
            r["ground_truth_label"] = int(gt)
    if "Attack" in df.columns:
        gt_attacks = df["Attack"].to_list()
        for r, atk in zip(results, gt_attacks):
            r["ground_truth_attack"] = str(atk)

    counts = {
        "benign_0": sum(1 for r in results if r["predicted_label"] == 0),
        "known_attack_1": sum(1 for r in results if r["predicted_label"] == 1),
        "unseen_alert_2": sum(1 for r in results if r["predicted_label"] == 2),
    }
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "filename": filename,
        "total_flows": len(results),
        "mc_samples_T": mc_samples,
        "std_threshold": std_threshold,
        "latency_ms": round(elapsed_ms, 2),
        "summary_counts": counts,
        "predictions": results,
    }


@app.get("/health")
def health_check():
    """Kiểm tra trạng thái sẵn sàng của Backend API và Mô hình."""
    return {
        "status": "healthy",
        "model_loaded": engine.model is not None,
        "model_path": MODEL_PATH,
        "device": str(engine.device),
        "feature_dim": EDGE_ATTR_DIM,
        "feature_names": FEATURE_COLS,
        "default_mc_samples": DEFAULT_MC_SAMPLES_T,
        "default_std_threshold": DEFAULT_STD_THRESHOLD,
        "demo_available": os.path.exists(DEMO_PARQUET_PATH),
    }


@app.get("/metrics")
def get_experiment_metrics():
    """Trả về toàn bộ các file kết quả thực nghiệm (.json) trong results/metrics/."""
    if not os.path.exists(METRICS_DIR):
        return {"metrics": {}}
    out: dict[str, Any] = {}
    for fname in sorted(os.listdir(METRICS_DIR)):
        if fname.endswith(".json"):
            fpath = os.path.join(METRICS_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    out[fname[:-5]] = json.load(f)
            except Exception:
                continue
    return {"metrics": out}


@app.post("/predict")
def predict_micro_batch(request: PredictBatchRequest):
    t0 = time.perf_counter()
    if not request.flows:
        raise HTTPException(status_code=400, detail="Danh sách 'flows' không được để trống.")

    src_nodes = []
    dst_nodes = []
    feat_rows = []

    for idx, item in enumerate(request.flows):
        src_nodes.append(item.src_node)
        dst_nodes.append(item.dst_node)

        if isinstance(item.features, list):
            if len(item.features) != EDGE_ATTR_DIM:
                raise HTTPException(
                    status_code=422,
                    detail=f"Flow thứ {idx} có {len(item.features)} đặc trưng, yêu cầu đúng {EDGE_ATTR_DIM} đặc trưng.",
                )
            feat_rows.append(item.features)
        elif isinstance(item.features, dict):
            row = [float(item.features.get(col, 0.0)) for col in FEATURE_COLS]
            feat_rows.append(row)
        else:
            raise HTTPException(
                status_code=422, detail=f"Định dạng 'features' ở flow {idx} không hợp lệ."
            )

    edge_features_np = np.asarray(feat_rows, dtype=np.float32)

    results = engine.run_three_way_inference(
        src_nodes=src_nodes,
        dst_nodes=dst_nodes,
        edge_features_np=edge_features_np,
        mc_samples=request.mc_samples,
        std_threshold=request.std_threshold,
    )

    counts = {
        "benign_0": sum(1 for r in results if r["predicted_label"] == 0),
        "known_attack_1": sum(1 for r in results if r["predicted_label"] == 1),
        "unseen_alert_2": sum(1 for r in results if r["predicted_label"] == 2),
    }
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "total_flows": len(results),
        "mc_samples_T": request.mc_samples,
        "std_threshold": request.std_threshold,
        "latency_ms": round(elapsed_ms, 2),
        "summary_counts": counts,
        "predictions": results,
    }


@app.post("/predict_demo")
def predict_from_builtin_demo(
    max_rows: int = Query(default=300, ge=10, le=1000, description="Số dòng đọc từ gói demo_microbatch"),
    mc_samples: int = Query(default=DEFAULT_MC_SAMPLES_T, ge=2, le=100),
    std_threshold: float = Query(default=DEFAULT_STD_THRESHOLD, gt=0.0, lt=1.0),
):
    """Chạy phân tích nhanh trên gói dữ liệu mẫu data/processed/demo_microbatch.parquet."""
    t0 = time.perf_counter()
    if not os.path.exists(DEMO_PARQUET_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy gói dữ liệu mẫu tại '{DEMO_PARQUET_PATH}'",
        )
    df = pl.read_parquet(DEMO_PARQUET_PATH).head(max_rows)
    return _format_dataframe_response(
        df=df,
        filename="demo_microbatch.parquet (50% Seen + 50% Unseen Bot & Infilteration)",
        mc_samples=mc_samples,
        std_threshold=std_threshold,
        t0=t0,
    )


@app.post("/predict_file")
async def predict_from_file(
    file: UploadFile = File(..., description="File CSV hoặc Parquet chứa micro-batch dữ liệu NetFlow"),
    max_rows: int = Query(default=500, ge=1, le=10000, description="Số dòng tối đa đọc từ file"),
    mc_samples: int = Query(default=DEFAULT_MC_SAMPLES_T, ge=2, le=100),
    std_threshold: float = Query(default=DEFAULT_STD_THRESHOLD, gt=0.0, lt=1.0),
):
    t0 = time.perf_counter()
    raw_bytes = await file.read()
    filename = (file.filename or "").lower()

    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(io.BytesIO(raw_bytes)).head(max_rows)
        elif filename.endswith(".csv"):
            df = pl.read_csv(io.BytesIO(raw_bytes), n_rows=max_rows)
        else:
            raise HTTPException(
                status_code=400, detail="Chỉ hỗ trợ định dạng file .csv hoặc .parquet"
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file: {exc}") from exc

    return _format_dataframe_response(
        df=df,
        filename=file.filename or "uploaded_file",
        mc_samples=mc_samples,
        std_threshold=std_threshold,
        t0=t0,
    )


WEB_DIST_DIR = os.path.join(os.path.dirname(__file__), "web", "dist")
WEB_ASSETS_DIR = os.path.join(WEB_DIST_DIR, "assets")
if os.path.exists(WEB_ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=WEB_ASSETS_DIR), name="web_assets")


@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#171923"/>'
        '<path d="M8 22V10l8 12V10m8 12V10" stroke="#e2a03f" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        "</svg>"
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/")
def serve_react_console():
    """Phục vụ giao diện React + TypeScript Console trực tiếp tại gốc http://127.0.0.1:8000/."""
    index_html = os.path.join(WEB_DIST_DIR, "index.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return {
        "message": "NF-UQ-NIDS-v2 API đang hoạt động. Truy cập /docs để xem Swagger UI hoặc build app/web để mở Console."
    }
