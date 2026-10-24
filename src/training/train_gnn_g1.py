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
from torch_geometric.nn import GINEConv

warnings.filterwarnings("ignore", category=FutureWarning)

# Đảm bảo console trên Windows hiển thị đúng tiếng Việt (UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


NUM_TOTAL_NODES = 284_339
NODE_INIT_DIM = 16
HIDDEN_DIM = 64
EDGE_ATTR_DIM = 41
BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 8192
NUM_NEIGHBORS = [10, 10]
NON_FEATURE_COLS = ["src_node_id", "dst_node_id", "label", "Attack", "Dataset"]


class GNNInductiveG1(nn.Module):
    """
    Kiến trúc G1 (Inductive Edge-Aware GNN với 2 lớp GINEConv):
    1. Khởi tạo Node Inductive tuyệt đối:
       - Loại bỏ hoàn toàn nn.Embedding.
       - Mọi node được khởi tạo bằng vector hằng số 16 chiều (`torch.ones((284339, 16))`),
         ép mô hình không được phép học thuộc lòng địa chỉ IP mà phải suy luận hoàn toàn
         từ topology mạng và đặc trưng giao tiếp NetFlow (`edge_attr`).
    2. Edge-Aware Message Passing (2 lớp GINEConv):
       - Đưa trực tiếp `edge_attr` (41 đặc trưng NetFlow) vào quá trình truyền thông điệp
         (cập nhật trạng thái nút nguồn & nút đích theo cả chiều đi và chiều về).
    3. Edge Classifier (MLP):
       - Nhận đầu vào là phép nối `[node_src || node_dst || edge_attr]` để dự đoán xác suất cạnh.
    """

    def __init__(
        self,
        node_init_dim: int = NODE_INIT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        edge_attr_dim: int = EDGE_ATTR_DIM,
    ):
        super().__init__()

        # Chiếu vector hằng số 16 chiều của Node lên không gian ẩn 64 chiều
        self.node_proj = nn.Linear(node_init_dim, hidden_dim)

        # Lớp GINEConv 1 (nhận node 64 chiều và edge_attr 41 chiều)
        gin_mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.conv1 = GINEConv(nn=gin_mlp1, edge_dim=edge_attr_dim, train_eps=True)

        # Lớp GINEConv 2 (nhận node 64 chiều và edge_attr 41 chiều)
        gin_mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.conv2 = GINEConv(nn=gin_mlp2, edge_dim=edge_attr_dim, train_eps=True)

        # Classifier (MLP) nhận đầu vào: [node_src (64) || node_dst (64) || edge_attr (41)] = 169 chiều
        classifier_in_dim = hidden_dim * 2 + edge_attr_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
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
        # 1. Chiếu đặc trưng hằng số ban đầu của node [num_sampled_nodes, 16] -> [num_sampled_nodes, 64]
        h = F.relu(self.node_proj(x))

        # 2. Truyền thông điệp hai chiều (Bidirectional Edge-Aware Message Passing) trên subgraph hiện tại:
        # Trong mạng máy tính (NetFlow), một IP nguồn phát tán tấn công (outbound edges) cần nhận được
        # thông điệp từ các luồng mà nó khởi tạo, đồng thời IP đích cũng nhận thông điệp từ các luồng đi tới.
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        bi_edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
        bi_edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        # Qua 2 lớp GINEConv có tích hợp edge_attr (41 chiều)
        h1 = self.conv1(h, bi_edge_index, bi_edge_attr)
        h1 = F.dropout(h1, p=0.2, training=self.training)
        h2 = self.conv2(h1, bi_edge_index, bi_edge_attr) + h1  # Residual connection

        # 3. Trích xuất biểu diễn của node nguồn và node đích cho các cạnh cần dự đoán
        src_idx, dst_idx = edge_label_index[0], edge_label_index[1]
        node_src = h2[src_idx]  # [batch_size, 64]
        node_dst = h2[dst_idx]  # [batch_size, 64]

        # 4. Phép nối [node_src, node_dst, edge_attr] -> [batch_size, 169]
        edge_repr = torch.cat([node_src, node_dst, target_edge_attr], dim=-1)

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
    - x         : torch.ones((284339, 16)) (chia sẻ bộ nhớ, không tốn thêm RAM)
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
    Đánh giá mô hình G1 không dùng gradient (torch.no_grad()):
    - Duyệt toàn bộ batch trong loader
    - Tính toán Precision, Recall, F1-score, PR-AUC.
    """
    model.eval()
    t0 = time.time()

    all_probs = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        # Đặc trưng của các cạnh mục tiêu cần dự đoán trong batch
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
    model_save_path = os.path.join(data_dir, "gine_g1_model.pt")

    train_max_edges = 4_096_000
    eval_seen_max_edges = 1_000_000

    # Khởi tạo Node Inductive tuyệt đối: Hằng số torch.ones((284339, 16))
    static_node_x = torch.ones((NUM_TOTAL_NODES, NODE_INIT_DIM), dtype=torch.float32)

    print("=" * 88)
    print("HUẤN LUYỆN & ĐÁNH GIÁ KIẾN TRÚC G1 (INDUCTIVE EDGE-AWARE GNN - 2 LỚP GINEConv)")
    print("=" * 88)
    print(f"- Thiết bị (Device)            : {device}")
    print(f"- Khởi tạo Node (Inductive)    : torch.ones(({NUM_TOTAL_NODES}, {NODE_INIT_DIM})) - Loại bỏ hoàn toàn nn.Embedding")
    print(f"- Message Passing              : 2 lớp GINEConv (edge_dim={EDGE_ATTR_DIM}, tích hợp 41 đặc trưng NetFlow)")
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

    print(f"[2/3] Đang huấn luyện mô hình G1 (GINEConv) trong 1 Epoch ({num_train_batches:,} batches)...")
    model = GNNInductiveG1(
        node_init_dim=NODE_INIT_DIM,
        hidden_dim=HIDDEN_DIM,
        edge_attr_dim=EDGE_ATTR_DIM,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
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

    # Lưu checkpoint mô hình G1 và giải phóng hoàn toàn đồ thị Train khỏi RAM
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_init_dim": NODE_INIT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "edge_attr_dim": EDGE_ATTR_DIM,
        },
        model_save_path,
    )
    print(f"      -> Huấn luyện 1 Epoch hoàn tất trong {time.time() - epoch_t0:.1f}s.")
    print(f"      -> Đã lưu checkpoint mô hình G1 tại: '{model_save_path}'\n")

    del train_loader, train_data
    gc.collect()

    # =========================================================================
    # 2. ĐÁNH GIÁ INDUCTIVE TRÊN 3 TẬP: VALIDATION, TEST_SEEN, TEST_UNSEEN
    #    (Mỗi tập tự tạo subgraph từ chính các luồng giao tiếp đang diễn ra,
    #     để GINEConv tổng hợp hành vi láng giềng thực tế của phiên mạng đó)
    # =========================================================================
    print("[3/3] Đang đánh giá Inductive trên 3 tập Validation, Test_seen và Test_unseen...")
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
    metrics_out_path = os.path.join(data_dir, "gnn_g1_metrics.json")
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # 3. IN BẢNG KẾT QUẢ & ĐỐI CHIẾU B0 (XGBOOST) vs G0 (GCN) vs G1 (GINEConv)
    # =========================================================================
    print("\n" + "=" * 88)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH G1 (INDUCTIVE GINEConv - CONSTANT NODE INIT)")
    print("=" * 88)
    print(
        f"{'Tập dữ liệu':<24} | {'Số cạnh Eval':>12} | {'Precision':>10} | {'Recall':>10} | {'F1-score':>10} | {'PR-AUC':>10}"
    )
    print("-" * 88)
    for m in results_summary:
        print(
            f"{m['split']:<24} | {m['rows']:>12,} | "
            f"{m['precision']:>10.4f} | {m['recall']:>10.4f} | "
            f"{m['f1']:>10.4f} | {m['pr_auc']:>10.4f}"
        )
    print("-" * 88)

    xgb_recall = 0.4571
    g0_recall = 0.1990
    g1_recall = tu_metrics["recall"]

    print("\n[ĐỐI CHIẾU TOÀN DIỆN TRÊN TẬP TEST_UNSEEN (ZERO-DAY: EXPLOITS & FUZZERS)]")
    print(f"1. B0 - XGBoost (Phi đồ thị)              : Recall = {xgb_recall*100:6.2f}% | F1 = 62.74% | PR-AUC (ghép Benign) = 0.9536")
    print(f"2. G0 - GCN (Transductive nn.Embedding)   : Recall = {g0_recall*100:6.2f}% | F1 = 33.20% | PR-AUC (ghép Benign) = 0.8898")
    print(
        f"3. G1 - GINEConv (Inductive Edge-Aware)   : Recall = {g1_recall*100:6.2f}% | "
        f"F1 = {tu_metrics['f1']*100:6.2f}% | PR-AUC (ghép Benign) = {tu_metrics['combined_with_benign_pr_auc']:.4f}"
    )
    print(
        f"-> Mức bứt phá của G1: {(g1_recall - xgb_recall)*100:+.2f}% so với B0 (XGBoost) | "
        f"{(g1_recall - g0_recall)*100:+.2f}% so với G0 (GCN)!"
    )
    print(f"-> Phát hiện đúng: {tu_metrics['TP']:,} / {tu_metrics['rows']:,} luồng Zero-day (Mean Prob = {tu_metrics['mean_pred_prob']:.4f})")
    print("- Chi tiết theo từng họ tấn công Zero-day (G1 - GINEConv):")
    for atk, info in tu_metrics["zero_day_breakdown"].items():
        print(
            f"   + {atk:<10}: {info['detected_tp']:>6,}/{info['count']:<6,} phát hiện được "
            f"| Recall = {info['recall']*100:>6.2f}% | Mean Prob = {info['mean_prob']:.4f}"
        )
    print("=" * 88)
    print(f"Tổng thời gian thực thi toàn bộ script: {time.time() - start_time:.1f} giây")


if __name__ == "__main__":
    main()
