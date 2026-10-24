import gc
import json
import os
import sys
import time
import warnings
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GATv2Conv

warnings.filterwarnings("ignore", category=FutureWarning)

# Đảm bảo console trên Windows hiển thị đúng tiếng Việt (UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


NUM_TOTAL_NODES = 284_339
NODE_INIT_DIM = 16
HIDDEN_DIM = 64
NUM_HEADS = 2
EDGE_ATTR_DIM = 41
BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 8192
NUM_NEIGHBORS = [10, 10]
POS_WEIGHT_VAL = 4.0  # Trọng số phạt lớp dương (Attack) để chống mất cân bằng miền UNSW-NB15
NON_FEATURE_COLS = ["src_node_id", "dst_node_id", "label", "Attack", "Dataset"]


class GNNInductiveGATv2(nn.Module):
    """
    Kiến trúc G1_v2 (Inductive Edge-Aware GATv2 + LayerNorm):
    1. Khởi tạo Node bằng hằng số `torch.ones((284339, 16))` (chống học thuộc lòng ID nút).
    2. 2 lớp `GATv2Conv(edge_dim=41)` đưa trực tiếp 41 đặc trưng NetFlow vào cơ chế Dynamic Attention,
       giúp mô hình tự động lọc các luồng láng giềng đáng ngờ thay vì cộng tổng (Sum Aggregation) bị lệch theo bậc nút.
    3. `LayerNorm` sau mỗi lớp GATv2Conv và trên `edge_attr` để ổn định phân phối biểu diễn giữa các miền dữ liệu.
    4. MLP Classifier kết hợp `[node_src || node_dst || edge_attr]` để dự đoán xác suất tấn công.
    """

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

        # Lớp GATv2Conv 1 (Multi-head Attention có tích hợp edge_dim=41)
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

        # Lớp GATv2Conv 2 (Multi-head Attention có tích hợp edge_dim=41)
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

        # Nhánh chiếu trực tiếp đặc trưng cạnh mục tiêu (Edge Residual Skip) kết hợp với biểu diễn Node
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_label_index: torch.Tensor,
        target_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Chiếu vector hằng số của node [N, 16] -> [N, 64]
        h = F.gelu(self.node_proj(x))

        # Chuẩn hóa nhẹ đặc trưng cạnh bằng LayerNorm để giảm Covariate Shift giữa các tập con
        e_sub = self.edge_norm(edge_attr)
        e_target = self.edge_norm(target_edge_attr)

        # Truyền thông điệp hai chiều để cả IP nguồn và IP đích đều cập nhật ngữ cảnh qua GATv2 Attention
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        bi_edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
        bi_edge_attr = torch.cat([e_sub, e_sub], dim=0)

        # 2. Qua 2 lớp GATv2Conv + LayerNorm + Residual
        h1 = self.gat1(h, bi_edge_index, edge_attr=bi_edge_attr)
        h1 = F.gelu(self.norm1(h1))

        h2 = self.gat2(h1, bi_edge_index, edge_attr=bi_edge_attr)
        h2 = F.gelu(self.norm2(h2 + h1))

        # 3. Trích xuất biểu diễn của node nguồn và node đích
        src_idx, dst_idx = edge_label_index[0], edge_label_index[1]
        node_src = h2[src_idx]  # [batch_size, 64]
        node_dst = h2[dst_idx]  # [batch_size, 64]

        # 4. Phép nối [node_src || node_dst || edge_attr] -> [batch_size, 169]
        edge_repr = torch.cat([node_src, node_dst, e_target], dim=-1)

        # 5. Dự đoán logit
        logits = self.classifier(edge_repr).squeeze(-1)
        return logits


def load_graph_split(
    parquet_path: str,
    static_node_x: torch.Tensor,
    max_edges: int | None = None,
) -> tuple[Data, int]:
    """
    Đọc file Parquet bằng Polars và đóng gói thành đối tượng PyG Data:
    - x         : torch.ones((284339, 16))
    - edge_index: [2, E] (int64)
    - edge_attr : [E, 41] (float32)
    - y         : [E] (float32)
    """
    lf = pl.scan_parquet(parquet_path)
    total_edges = lf.select(pl.len()).collect(engine="streaming").item()
    schema_cols = lf.collect_schema().names()
    feature_cols = [c for c in schema_cols if c not in NON_FEATURE_COLS]

    if max_edges is not None and total_edges > max_edges:
        lf = lf.head(max_edges)

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

    data = Data(
        x=static_node_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )
    return data, total_edges


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: LinkNeighborLoader,
    full_target_edge_attr: torch.Tensor,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Đánh giá mô hình G1_v2 không dùng gradient (torch.no_grad()):
    - Duyệt toàn bộ batch trong loader
    - Tính toán Precision, Recall, F1-score, PR-AUC.
    """
    model.eval()
    t0 = time.time()

    all_probs = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        target_edge_attr = full_target_edge_attr[batch.input_id.cpu()].to(device)

        logits = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            edge_label_index=batch.edge_label_index,
            target_edge_attr=target_edge_attr,
        )
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        labels = batch.edge_label.cpu().numpy().astype(np.int8)

        all_probs.append(probs)
        all_labels.append(labels)

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = (y_prob >= 0.5).astype(np.int8)
    del all_probs, all_labels
    gc.collect()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    pr_auc = float(average_precision_score(y_true, y_prob))

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    metrics = {
        "rows": int(len(y_true)),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "mean_pred_prob": float(np.mean(y_prob)),
        "eval_time_sec": round(time.time() - t0, 2),
    }
    return metrics, y_true, y_prob


def main():
    start_time = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data"
    train_parquet = os.path.join(data_dir, "edges_train.parquet")
    val_parquet = os.path.join(data_dir, "edges_val.parquet")
    test_seen_parquet = os.path.join(data_dir, "edges_test_seen.parquet")
    test_unseen_parquet = os.path.join(data_dir, "edges_test_unseen.parquet")
    model_save_path = os.path.join(data_dir, "gatv2_g1_v2_model.pt")

    train_max_edges = 4_096_000
    eval_seen_max_edges = 1_000_000

    # Khởi tạo Node Inductive bằng hằng số torch.ones((284339, 16))
    static_node_x = torch.ones((NUM_TOTAL_NODES, NODE_INIT_DIM), dtype=torch.float32)

    print("=" * 90)
    print("HUẤN LUYỆN & ĐÁNH GIÁ G1_v2 (INDUCTIVE GATv2Conv + LAYERNORM + POS_WEIGHT RE-WEIGHTING)")
    print("=" * 90)
    print(f"- Thiết bị (Device)            : {device}")
    print(f"- Khởi tạo Node (Inductive)    : torch.ones(({NUM_TOTAL_NODES}, {NODE_INIT_DIM})) - Không dùng nn.Embedding")
    print(f"- Kiến trúc Message Passing    : 2 lớp GATv2Conv (heads={NUM_HEADS}, edge_dim={EDGE_ATTR_DIM}) + LayerNorm")
    print(f"- Xử lý mất cân bằng miền/lớp  : BCEWithLogitsLoss(pos_weight={POS_WEIGHT_VAL:.1f})")
    print(f"- Cấu hình LinkNeighborLoader  : num_neighbors={NUM_NEIGHBORS}, batch_size={BATCH_SIZE}, shuffle=True\n")

    # =========================================================================
    # 1. NẠP ĐỒ THỊ TRAIN & HUẤN LUYỆN 1 EPOCH
    # =========================================================================
    print("[1/3] Đang đọc 'data/edges_train.parquet' và khởi tạo đồ thị Train...")
    t0 = time.time()
    train_data, total_train_rows = load_graph_split(
        train_parquet, static_node_x, max_edges=train_max_edges
    )
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        edge_label_index=train_data.edge_index,
        edge_label=train_data.y,
        shuffle=True,
    )
    num_train_batches = len(train_loader)
    print(
        f"      -> Đã nạp đồ thị Train: {train_data.edge_index.size(1):,}/{total_train_rows:,} cạnh "
        f"({num_train_batches:,} batches) trong {time.time() - t0:.1f}s.\n"
    )

    print(f"[2/3] Đang huấn luyện mô hình G1_v2 (GATv2Conv + LayerNorm) trong 1 Epoch ({num_train_batches:,} batches)...")
    model = GNNInductiveGATv2(
        node_init_dim=NODE_INIT_DIM,
        hidden_dim=HIDDEN_DIM,
        heads=NUM_HEADS,
        edge_attr_dim=EDGE_ATTR_DIM,
    ).to(device)

    pos_weight_tensor = torch.tensor([POS_WEIGHT_VAL], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_batches)

    model.train()
    epoch_t0 = time.time()
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        target_edge_attr = train_data.edge_attr[batch.input_id.cpu()].to(device)
        target_labels = batch.edge_label.float()

        logits = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            edge_label_index=batch.edge_label_index,
            target_edge_attr=target_edge_attr,
        )
        loss = criterion(logits, target_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == num_train_batches:
            avg_loss = running_loss / (batch_idx + 1)
            elapsed = time.time() - epoch_t0
            print(
                f"      -> Epoch 1 | Batch {batch_idx + 1:>4}/{num_train_batches} | "
                f"Current Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_init_dim": NODE_INIT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "heads": NUM_HEADS,
            "edge_attr_dim": EDGE_ATTR_DIM,
            "pos_weight": POS_WEIGHT_VAL,
        },
        model_save_path,
    )
    print(f"      -> Huấn luyện 1 Epoch hoàn tất trong {time.time() - epoch_t0:.1f}s.")
    print(f"      -> Đã lưu checkpoint mô hình G1_v2 tại: '{model_save_path}'\n")

    del train_loader, train_data
    gc.collect()

    # =========================================================================
    # 2. ĐÁNH GIÁ TRÊN 3 TẬP: VALIDATION, TEST_SEEN, TEST_UNSEEN
    # =========================================================================
    print("[3/3] Đang đánh giá trên 3 tập Validation, Test_seen và Test_unseen...")
    results_summary = []

    # --- 2.1. Validation ---
    print("      -> (1/3) Đang đánh giá trên tập Validation ('data/edges_val.parquet')...")
    val_data, total_val = load_graph_split(val_parquet, static_node_x, max_edges=eval_seen_max_edges)
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=EVAL_BATCH_SIZE,
        edge_label_index=val_data.edge_index,
        edge_label=val_data.y,
        shuffle=False,
    )
    val_metrics, _, _ = evaluate_model(model, val_loader, val_data.edge_attr, device)
    val_metrics["split"] = "Validation"
    val_metrics["total_file_rows"] = total_val
    results_summary.append(val_metrics)
    del val_loader, val_data
    gc.collect()
    print(
        f"         + Hoàn tất Validation ({val_metrics['rows']:,} cạnh) trong {val_metrics['eval_time_sec']}s: "
        f"F1={val_metrics['f1']:.4f}, Recall={val_metrics['recall']:.4f}, PR-AUC={val_metrics['pr_auc']:.4f}"
    )

    # --- 2.2. Test_seen ---
    print("      -> (2/3) Đang đánh giá trên tập Test_seen ('data/edges_test_seen.parquet')...")
    ts_data, total_ts = load_graph_split(test_seen_parquet, static_node_x, max_edges=eval_seen_max_edges)
    ts_loader = LinkNeighborLoader(
        data=ts_data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=EVAL_BATCH_SIZE,
        edge_label_index=ts_data.edge_index,
        edge_label=ts_data.y,
        shuffle=False,
    )
    ts_metrics, ts_y_true, ts_y_prob = evaluate_model(model, ts_loader, ts_data.edge_attr, device)
    ts_metrics["split"] = "Test_seen"
    ts_metrics["total_file_rows"] = total_ts
    results_summary.append(ts_metrics)

    benign_probs_ref = ts_y_prob[ts_y_true == 0]
    del ts_loader, ts_data, ts_y_true, ts_y_prob
    gc.collect()
    print(
        f"         + Hoàn tất Test_seen ({ts_metrics['rows']:,} cạnh) trong {ts_metrics['eval_time_sec']}s: "
        f"F1={ts_metrics['f1']:.4f}, Recall={ts_metrics['recall']:.4f}, PR-AUC={ts_metrics['pr_auc']:.4f}"
    )

    # --- 2.3. Test_unseen (Zero-day: TOÀN BỘ 53,861 cạnh Exploits & Fuzzers) ---
    print("      -> (3/3) Đang đánh giá trên TOÀN BỘ tập Test_unseen ('data/edges_test_unseen.parquet')...")
    tu_data, total_tu = load_graph_split(test_unseen_parquet, static_node_x, max_edges=None)
    tu_loader = LinkNeighborLoader(
        data=tu_data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=EVAL_BATCH_SIZE,
        edge_label_index=tu_data.edge_index,
        edge_label=tu_data.y,
        shuffle=False,
    )
    tu_metrics, tu_y_true, tu_y_prob = evaluate_model(model, tu_loader, tu_data.edge_attr, device)
    tu_metrics["split"] = "Test_unseen (Zero-day)"
    tu_metrics["total_file_rows"] = total_tu

    # Tính PR-AUC khi ghép cân bằng 1:1 với mẫu Benign từ Test_seen
    n_unseen = len(tu_y_true)
    ben_sample = benign_probs_ref[:n_unseen]
    y_comb = np.concatenate([tu_y_true, np.zeros(len(ben_sample), dtype=np.int8)])
    prob_comb = np.concatenate([tu_y_prob, ben_sample])
    pred_comb = (prob_comb >= 0.5).astype(np.int8)

    tu_metrics["combined_with_benign_precision"] = float(precision_score(y_comb, pred_comb, zero_division=0))
    tu_metrics["combined_with_benign_f1"] = float(f1_score(y_comb, pred_comb, zero_division=0))
    tu_metrics["combined_with_benign_pr_auc"] = float(average_precision_score(y_comb, prob_comb))

    # Phân tích chi tiết theo từng họ tấn công Zero-day (Exploits vs Fuzzers)
    attacks = (
        pl.scan_parquet(test_unseen_parquet)
        .select("Attack")
        .collect(engine="streaming")["Attack"]
        .to_list()
    )
    attack_arr = np.array(attacks)
    tu_pred = (tu_y_prob >= 0.5).astype(np.int8)
    per_attack_recall = {}
    for atk in np.unique(attack_arr):
        mask = attack_arr == atk
        per_attack_recall[str(atk)] = {
            "count": int(mask.sum()),
            "detected_tp": int(tu_pred[mask].sum()),
            "recall": float(tu_pred[mask].mean()),
            "mean_prob": float(tu_y_prob[mask].mean()),
        }
    tu_metrics["zero_day_breakdown"] = per_attack_recall
    results_summary.append(tu_metrics)

    del tu_loader, tu_data, tu_y_true, tu_y_prob, benign_probs_ref
    gc.collect()

    # Lưu kết quả ra file JSON
    metrics_out_path = os.path.join(data_dir, "gnn_g1_v2_metrics.json")
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # 3. IN BẢNG KẾT QUẢ & ĐỐI CHIẾU TOÀN DIỆN (B0 vs G0 vs G1 vs G1_v2)
    # =========================================================================
    print("\n" + "=" * 90)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH G1_v2 (GATv2Conv + LAYERNORM + POS_WEIGHT=4.0)")
    print("=" * 90)
    print(
        f"{'Tập dữ liệu':<24} | {'Số cạnh Eval':>12} | {'Precision':>10} | {'Recall':>10} | {'F1-score':>10} | {'PR-AUC':>10}"
    )
    print("-" * 90)
    for m in results_summary:
        print(
            f"{m['split']:<24} | {m['rows']:>12,} | "
            f"{m['precision']:>10.4f} | {m['recall']:>10.4f} | "
            f"{m['f1']:>10.4f} | {m['pr_auc']:>10.4f}"
        )
    print("-" * 90)

    xgb_recall = 0.4571
    g0_recall = 0.1990
    g1_recall = 0.0584
    g1v2_recall = tu_metrics["recall"]

    print("\n[ĐỐI CHIẾU TRỰC TIẾP TRÊN TẬP TEST_UNSEEN (ZERO-DAY: EXPLOITS & FUZZERS)]")
    print(f"1. B0    - XGBoost (Phi đồ thị)            : Recall = {xgb_recall*100:6.2f}% | F1 = 62.74% | PR-AUC (ghép Benign) = 0.9536")
    print(f"2. G0    - GCN (Transductive nn.Embedding) : Recall = {g0_recall*100:6.2f}% | F1 = 33.20% | PR-AUC (ghép Benign) = 0.8898")
    print(f"3. G1    - GINEConv (Sum Aggregation)      : Recall = {g1_recall*100:6.2f}% | F1 = 11.03% | PR-AUC (ghép Benign) = 0.6523")
    print(
        f"4. G1_v2 - GATv2Conv + LayerNorm + Reweight: Recall = {g1v2_recall*100:6.2f}% | "
        f"F1 = {tu_metrics['f1']*100:6.2f}% | PR-AUC (ghép Benign) = {tu_metrics['combined_with_benign_pr_auc']:.4f}"
    )
    print(
        f"-> So sánh G1_v2 với XGBoost (B0): {(g1v2_recall - xgb_recall)*100:+.2f}% Recall "
        f"({tu_metrics['TP']:,} / {tu_metrics['rows']:,} luồng Zero-day được phát hiện, Mean Prob = {tu_metrics['mean_pred_prob']:.4f})"
    )
    print("- Chi tiết theo từng họ tấn công Zero-day (G1_v2 - GATv2Conv):")
    for atk, info in tu_metrics["zero_day_breakdown"].items():
        print(
            f"   + {atk:<10}: {info['detected_tp']:>6,}/{info['count']:<6,} phát hiện được "
            f"| Recall = {info['recall']*100:>6.2f}% | Mean Prob = {info['mean_prob']:.4f}"
        )
    print("=" * 90)
    print(f"Tổng thời gian thực thi toàn bộ script: {time.time() - start_time:.1f} giây")


if __name__ == "__main__":
    main()
