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
import numpy as np
import polars as pl
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 1. CẤU HÌNH HỆ THỐNG & ĐẶC TRƯNG NETFLOW (41 FEATURES)
# =============================================================================
MODEL_PATH = os.path.join("models", "gnns", "gatv2_g1_v2_model.pt")
PREPROC_PATH = os.path.join("data", "processed", "preprocessor_params.json")

NODE_INIT_DIM = 16
HIDDEN_DIM = 64
NUM_HEADS = 2
EDGE_ATTR_DIM = 41
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
    2: "Zero-day Alert (High Epistemic Uncertainty)",
}


# =============================================================================
# 2. KIẾN TRÚC MÔ HÌNH GATv2 (G1_v2) TÍCH HỢP MC DROPOUT
# =============================================================================
class GNNInductiveGATv2(nn.Module):
    """Kiến trúc G1_v2 (Inductive Edge-Aware GATv2 + LayerNorm) hỗ trợ Three-way MC Dropout UQ."""

    def __init__(
        self,
        node_init_dim: int = NODE_INIT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        heads: int = NUM_HEADS,
        edge_attr_dim: int = EDGE_ATTR_DIM,
    ):
        super().__init__()
        out_per_head = hidden_dim // heads

        self.node_proj = nn.Linear(node_init_dim, hidden_dim)
        self.edge_norm = nn.LayerNorm(edge_attr_dim)

        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=out_per_head,
            heads=heads,
            concat=True,
            edge_dim=edge_attr_dim,
            dropout=0.1,
            add_self_loops=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=out_per_head,
            heads=heads,
            concat=True,
            edge_dim=edge_attr_dim,
            dropout=0.1,
            add_self_loops=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.mc_node_drop = nn.Dropout(p=0.15)

        classifier_in_dim = hidden_dim * 2 + edge_attr_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_subgraph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.gelu(self.node_proj(x))
        e_norm = self.edge_norm(edge_attr)

        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        bi_edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
        bi_edge_attr = torch.cat([e_norm, e_norm], dim=0)
        return h, bi_edge_index, bi_edge_attr, e_norm

    def forward_single_mc(
        self,
        h: torch.Tensor,
        bi_edge_index: torch.Tensor,
        bi_edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        e_norm: torch.Tensor,
    ) -> torch.Tensor:
        h1 = self.gat1(h, bi_edge_index, edge_attr=bi_edge_attr)
        h1 = F.gelu(self.norm1(h1))
        h1 = self.mc_node_drop(h1)

        h2 = self.gat2(h1, bi_edge_index, edge_attr=bi_edge_attr)
        h2 = F.gelu(self.norm2(h2 + h1))
        h2 = self.mc_node_drop(h2)

        node_src = h2[edge_index[0]]
        node_dst = h2[edge_index[1]]

        edge_repr = torch.cat([node_src, node_dst, e_norm], dim=-1)
        logits = self.classifier(edge_repr).squeeze(-1)
        return torch.sigmoid(logits)


# =============================================================================
# 3. QUẢN LÝ TRẠNG THÁI MODEL TRONG BỘ NHỚ (LIFESPAN / STARTUP)
# =============================================================================
class NIDSRuntimeEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: GNNInductiveGATv2 | None = None
        self.preproc_params: dict[str, Any] | None = None
        self.bg_edge_index: torch.Tensor | None = None
        self.bg_edge_attr: torch.Tensor | None = None
        self.bg_num_nodes: int = 284_339
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

        # Nạp sẵn một cửa sổ đồ thị ngữ cảnh nền (50,000 cạnh) vào RAM lúc khởi động để khi micro-batch nhỏ (1..50 dòng)
        # được gửi qua API, các node vẫn có đủ láng giềng thực tế cho 2 lớp GATv2Conv truyền thông điệp.
        bg_parquet = os.path.join("data", "processed", "edges_test_seen.parquet")
        if os.path.exists(bg_parquet):
            df_bg = pl.scan_parquet(bg_parquet).head(50_000).collect(engine="streaming")
            s_np = df_bg["src_node_id"].to_numpy().astype(np.int64, copy=False)
            d_np = df_bg["dst_node_id"].to_numpy().astype(np.int64, copy=False)
            self.bg_edge_index = torch.from_numpy(np.vstack([s_np, d_np])).to(self.device)
            self.bg_edge_attr = (
                torch.from_numpy(
                    df_bg.select([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS]).to_numpy()
                ).to(self.device)
            )
            del df_bg

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
        """
        Dựng đồ thị con Inductive tức thời từ micro-batch các luồng NetFlow (kết hợp láng giềng ngữ cảnh),
        chạy T=20 lần MC Dropout và phân loại theo quy tắc Three-way UQ:
          - 0: Benign (Mean Prob < 0.5 & Std <= std_threshold)
          - 1: Known Attack (Mean Prob >= 0.5 & Std <= std_threshold)
          - 2: Zero-day Alert (Std > std_threshold)
        """
        if self.model is None:
            raise RuntimeError("Mô hình chưa được khởi tạo trong bộ nhớ!")

        num_flows = len(src_nodes)
        if num_flows == 0:
            return []

        # Chuyển src_nodes / dst_nodes sang chỉ số node nguyên (nếu là chuỗi IP mới thì băm về miền ID)
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

        # Lấy thêm các cạnh láng giềng 1-hop từ đồ thị nền (nếu có) liên kết với các node trong micro-batch
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

        # Re-index các node xuất hiện trong subgraph về 0..K-1 để tiết kiệm bộ nhớ và chạy siêu tốc
        unique_nodes, relabeled = torch.unique(full_ei, return_inverse=True)
        local_full_ei = relabeled.view(2, -1)
        local_target_ei = local_full_ei[:, :num_flows]

        x = torch.ones((unique_nodes.size(0), NODE_INIT_DIM), dtype=torch.float32, device=self.device)

        # Tiền mã hóa biểu diễn chuẩn hóa
        self.model.eval()
        h, bi_ei, bi_ea, e_norm = self.model.encode_subgraph(x, local_full_ei, full_ea)
        target_e_norm = e_norm[:num_flows]

        # Bật MC Dropout (model.train() trong torch.no_grad()) và lấy mẫu T=20 lần
        self.model.train()
        mc_preds = []
        for _ in range(mc_samples):
            prob_t = self.model.forward_single_mc(h, bi_ei, bi_ea, local_target_ei, target_e_norm)
            mc_preds.append(prob_t.view(-1))

        mc_stack = torch.stack(mc_preds, dim=0)  # [T, num_flows]
        mean_probs = mc_stack.mean(dim=0).cpu().numpy()
        epistemic_stds = mc_stack.std(dim=0, unbiased=False).cpu().numpy()

        predictions = []
        for idx in range(num_flows):
            m_prob = float(mean_probs[idx])
            e_std = float(epistemic_stds[idx])

            # Quy tắc Three-way UQ:
            if e_std > std_threshold:
                pred_code = 2  # Zero-day Alert (High Epistemic Uncertainty)
            elif m_prob >= 0.5:
                pred_code = 1  # Known Attack
            else:
                pred_code = 0  # Trusted Benign

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
    title="NF-UQ-NIDS-v2 Zero-day Detection API",
    description=(
        "Backend API phát hiện xâm nhập mạng và tấn công Zero-day sử dụng kiến trúc "
        "Inductive Edge-Aware GATv2 kết hợp Định lượng bất định 3 nhánh (Three-way MC Dropout UQ)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 4. SCHEMAS CHO REQUEST & RESPONSE JSON
# =============================================================================
class FlowItem(BaseModel):
    src_node: str | int = Field(default="0", description="Địa chỉ IP nguồn hoặc src_node_id")
    dst_node: str | int = Field(default="1", description="Địa chỉ IP đích hoặc dst_node_id")
    features: list[float] | dict[str, float] = Field(
        ...,
        description="Danh sách 41 đặc trưng NetFlow đã chuẩn hóa (list 41 số thực) hoặc dict tên cột -> giá trị",
    )


class PredictBatchRequest(BaseModel):
    flows: list[FlowItem] = Field(..., description="Danh sách các luồng NetFlow trong micro-batch")
    mc_samples: int = Field(default=DEFAULT_MC_SAMPLES_T, ge=2, le=100, description="Số lần lấy mẫu MC Dropout (T)")
    std_threshold: float = Field(default=DEFAULT_STD_THRESHOLD, gt=0.0, lt=1.0, description="Ngưỡng Epistemic Std phát hiện Zero-day")


# =============================================================================
# 5. CÁC ENDPOINTS CỦA HỆ THỐNG
# =============================================================================
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
    }


@app.post("/predict")
def predict_micro_batch(request: PredictBatchRequest):
    """
    Endpoint chính `/predict`:
    Nhận một micro-batch các luồng giao tiếp NetFlow (JSON), chạy suy luận qua mô hình GATv2
    với cơ chế Three-way UQ (MC Dropout T=20) và trả về:
      - `predicted_label`: 0 (Benign), 1 (Known Attack), 2 (Zero-day Alert)
      - `mean_prob`: Xác suất dự đoán trung bình
      - `epistemic_std`: Độ bất định nhận thức (Epistemic Std)
    """
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
            raise HTTPException(status_code=422, detail=f"Định dạng 'features' ở flow {idx} không hợp lệ.")

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
        "zeroday_alert_2": sum(1 for r in results if r["predicted_label"] == 2),
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


@app.post("/predict_file")
async def predict_from_file(
    file: UploadFile = File(..., description="File CSV hoặc Parquet chứa micro-batch dữ liệu NetFlow"),
    max_rows: int = Query(default=500, ge=1, le=10000, description="Số dòng tối đa đọc từ file"),
    mc_samples: int = Query(default=DEFAULT_MC_SAMPLES_T, ge=2, le=100),
    std_threshold: float = Query(default=DEFAULT_STD_THRESHOLD, gt=0.0, lt=1.0),
):
    """
    Endpoint hỗ trợ tải lên trực tiếp file CSV hoặc Parquet nhỏ (dành cho Dashboard upload file),
    tự động trích xuất 41 cột đặc trưng và trả về kết quả phân loại Three-way UQ.
    """
    t0 = time.perf_counter()
    raw_bytes = await file.read()
    filename = (file.filename or "").lower()

    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(io.BytesIO(raw_bytes)).head(max_rows)
        elif filename.endswith(".csv"):
            df = pl.read_csv(io.BytesIO(raw_bytes), n_rows=max_rows)
        else:
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ định dạng file .csv hoặc .parquet")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file: {exc}") from exc

    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"File thiếu các cột đặc trưng bắt buộc: {missing_cols[:5]}...",
        )

    if "src_node_id" in df.columns and "dst_node_id" in df.columns:
        src_nodes = df["src_node_id"].to_list()
        dst_nodes = df["dst_node_id"].to_list()
    elif "IPV4_SRC_ADDR" in df.columns and "IPV4_DST_ADDR" in df.columns:
        src_nodes = df["IPV4_SRC_ADDR"].to_list()
        dst_nodes = df["IPV4_DST_ADDR"].to_list()
    else:
        src_nodes = list(range(df.height))
        dst_nodes = [i + 1 for i in range(df.height)]

    edge_features_np = (
        df.select([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS])
        .to_numpy()
    )

    results = engine.run_three_way_inference(
        src_nodes=src_nodes,
        dst_nodes=dst_nodes,
        edge_features_np=edge_features_np,
        mc_samples=mc_samples,
        std_threshold=std_threshold,
    )

    # Nếu file có cột nhãn thực tế (label / Attack), đính kèm vào kết quả để Dashboard đối chiếu
    if "label" in df.columns:
        gt_labels = df["label"].to_list()
        for r, gt in zip(results, gt_labels):
            r["ground_truth_label"] = int(gt)
    if "Attack" in df.columns:
        gt_attacks = df["Attack"].to_list()
        for r, atk in zip(results, gt_attacks):
            r["ground_truth_attack"] = str(atk)

    counts = {
        "benign_0": sum(1 for r in results if r["predicted_label"] == 0),
        "known_attack_1": sum(1 for r in results if r["predicted_label"] == 1),
        "zeroday_alert_2": sum(1 for r in results if r["predicted_label"] == 2),
    }
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "filename": file.filename,
        "total_flows": len(results),
        "mc_samples_T": mc_samples,
        "std_threshold": std_threshold,
        "latency_ms": round(elapsed_ms, 2),
        "summary_counts": counts,
        "predictions": results,
    }
