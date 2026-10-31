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
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.gnn_architectures import (
    EDGE_ATTR_DIM,
    HIDDEN_DIM,
    NON_FEATURE_COLS,
    NUM_TOTAL_NODES,
    GNNBaselineG0,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Đảm bảo console trên Windows hiển thị đúng tiếng Việt (UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


EMBEDDING_DIM = 64
BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 8192
NUM_NEIGHBORS = [10, 10]


def load_edge_tensors(
    parquet_path: str, max_edges: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Đọc file Parquet bằng Polars và chuyển thành các Tensor:
    - edge_index: [2, E] (torch.int64)
    - edge_attr : [E, 41] (torch.float32)
    - y         : [E] (torch.float32)
    """
    lf = pl.scan_parquet(parquet_path)
    total_edges = lf.select(pl.len()).collect(engine="streaming").item()
    schema_cols = lf.collect_schema().names()
    feature_cols = [c for c in schema_cols if c not in NON_FEATURE_COLS]

    if max_edges is not None and total_edges > max_edges:
        lf = lf.head(max_edges)
        loaded_edges = max_edges
    else:
        loaded_edges = total_edges

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

    return edge_index, edge_attr, y, total_edges


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: LinkNeighborLoader,
    target_edge_attr_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Hàm đánh giá mô hình không dùng gradient (torch.no_grad()):
    - Duyệt qua toàn bộ batch trong loader
    - Thu thập xác suất dự đoán (Sigmoid) và nhãn thực tế
    - Tính toán 4 chỉ số: Precision, Recall, F1-score, và PR-AUC (scikit-learn).
    """
    model.eval()
    t0 = time.time()

    all_probs = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        # Lấy đặc trưng cạnh của đúng các cạnh mục tiêu (edge_label_index) thông qua batch.input_id
        batch_edge_attr = target_edge_attr_tensor[batch.input_id.cpu()].to(device)

        logits = model(
            n_id=batch.n_id,
            edge_index=batch.edge_index,
            edge_label_index=batch.edge_label_index,
            target_edge_attr=batch_edge_attr,
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


def build_eval_loader(
    base_train_graph: Data,
    eval_edge_index: torch.Tensor,
    eval_y: torch.Tensor,
) -> LinkNeighborLoader:
    """
    Cấu hình LinkNeighborLoader chống rò rỉ dữ liệu (Data Leakage Prevention):
    - `data=base_train_graph`: Cạnh dùng để truyền thông điệp (message passing `edge_index`)
      được lấy hoàn toàn từ đồ thị Train.
    - `edge_label_index=eval_edge_index`: Các cạnh cần dự đoán thuộc tập Val / Test_seen / Test_unseen.
    """
    return LinkNeighborLoader(
        data=base_train_graph,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=EVAL_BATCH_SIZE,
        edge_label_index=eval_edge_index,
        edge_label=eval_y,
        shuffle=False,
    )


def main():
    start_time = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = os.path.join("data", "processed")
    train_parquet = os.path.join(proc_dir, "edges_train.parquet")
    val_parquet = os.path.join(proc_dir, "edges_val.parquet")
    test_seen_parquet = os.path.join(proc_dir, "edges_test_seen.parquet")
    test_unseen_parquet = os.path.join(proc_dir, "edges_test_unseen.parquet")
    os.makedirs(os.path.join("models", "gnns"), exist_ok=True)
    model_save_path = os.path.join("models", "gnns", "gcn_g0_model.pt")

    # Để đảm bảo thời gian 1 Epoch của LinkNeighborLoader trên CPU khả thi (~2-3 phút) và không bị OOM
    # khi giữ đồ thị truyền thông điệp trong RAM, ta nạp 4,096,000 cạnh Train (1,000 mini-batches x 4096)
    # đã được xáo trộn ngẫu nhiên (seed=42) cho đồ thị Train, 1,000,000 cạnh cho Val & Test_seen,
    # và TOÀN BỘ 259,458 cạnh của Test_unseen (Bot & Infilteration).
    train_max_edges = 4_096_000
    eval_seen_max_edges = 1_000_000

    print("=" * 88)
    print("HUẤN LUYỆN & ĐÁNH GIÁ TOÀN DIỆN GNN BASELINE (G0 - GCN) CHO PHÁT HIỆN UNSEEN ATTACK")
    print("=" * 88)
    print(f"- Thiết bị (Device)            : {device}")
    print(f"- Tổng số Node toàn mạng       : {NUM_TOTAL_NODES:,} (nn.Embedding: {NUM_TOTAL_NODES} x {EMBEDDING_DIM})")
    print(f"- Cấu hình LinkNeighborLoader  : num_neighbors={NUM_NEIGHBORS}, train_batch_size={BATCH_SIZE}")
    print(f"- Chống Data Leakage Inference : Message Passing trên đồ thị Train, dự đoán trên cạnh Val/Test\n")

    # =========================================================================
    # 1. KHỞI TẠO ĐỒ THỊ TRAIN & LINKNEIGHBORLOADER
    # =========================================================================
    print("[1/4] Đang đọc 'data/processed/edges_train.parquet' và khởi tạo đồ thị Train...")
    t0 = time.time()
    train_edge_index, train_edge_attr, train_y, total_train_rows = load_edge_tensors(
        train_parquet, max_edges=train_max_edges
    )

    # Đối tượng Data chỉ giữ cấu trúc đồ thị (edge_index) cho Message Passing nhẹ nhàng,
    # còn train_edge_attr được truy xuất trực tiếp qua batch.input_id
    train_graph = Data(
        num_nodes=NUM_TOTAL_NODES,
        edge_index=train_edge_index,
        y=train_y,
    )
    train_loader = LinkNeighborLoader(
        data=train_graph,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        edge_label_index=train_graph.edge_index,
        edge_label=train_graph.y,
        shuffle=True,
    )
    num_train_batches = len(train_loader)
    print(
        f"      -> Đã nạp đồ thị Train: {train_edge_index.size(1):,}/{total_train_rows:,} cạnh "
        f"({num_train_batches:,} batches) trong {time.time() - t0:.1f}s.\n"
    )

    # =========================================================================
    # 2. HUẤN LUYỆN 1 EPOCH TRÊN TẬP TRAIN & LƯU MODEL
    # =========================================================================
    print(f"[2/4] Đang huấn luyện GNN Baseline (G0 - GCN) trong 1 Epoch ({num_train_batches:,} batches)...")
    model = GNNBaselineG0(
        num_nodes=NUM_TOTAL_NODES,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        edge_attr_dim=EDGE_ATTR_DIM,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    model.train()
    epoch_t0 = time.time()
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        target_edge_attr = train_edge_attr[batch.input_id.cpu()].to(device)
        target_labels = batch.edge_label.float()

        logits = model(
            n_id=batch.n_id,
            edge_index=batch.edge_index,
            edge_label_index=batch.edge_label_index,
            target_edge_attr=target_edge_attr,
        )
        loss = criterion(logits, target_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == num_train_batches:
            avg_loss = running_loss / (batch_idx + 1)
            elapsed = time.time() - epoch_t0
            print(
                f"      -> Epoch 1 | Batch {batch_idx + 1:>4}/{num_train_batches} | "
                f"Current Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    # Giải phóng train_edge_attr và train_loader sau khi huấn luyện xong, chỉ giữ lại train_graph (edge_index)
    # để làm đồ thị nền truyền thông điệp (Message Passing) cho bước Inference
    del train_loader, train_edge_attr, train_y
    train_graph.y = None
    gc.collect()

    # Lưu checkpoint mô hình
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_nodes": NUM_TOTAL_NODES,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "edge_attr_dim": EDGE_ATTR_DIM,
        },
        model_save_path,
    )
    print(f"      -> Huấn luyện 1 Epoch hoàn tất trong {time.time() - epoch_t0:.1f}s.")
    print(f"      -> Đã lưu checkpoint mô hình tại: '{model_save_path}'\n")

    # =========================================================================
    # 3. ĐÁNH GIÁ TRÊN 3 TẬP: VALIDATION, TEST_SEEN, TEST_UNSEEN
    #    (Dùng edge_index của Train để truyền thông điệp, chống Data Leakage)
    # =========================================================================
    print("[3/4] Đang cấu hình 3 LinkNeighborLoader riêng biệt và đánh giá (Chống Data Leakage)...")
    results_summary = []

    # --- 3.1. Validation ---
    print("      -> (1/3) Đang đánh giá trên tập Validation ('data/processed/edges_val.parquet')...")
    val_ei, val_ea, val_y, total_val = load_edge_tensors(val_parquet, max_edges=eval_seen_max_edges)
    val_loader = build_eval_loader(train_graph, val_ei, val_y)
    val_metrics, _, _ = evaluate_model(model, val_loader, val_ea, device)
    val_metrics["split"] = "Validation"
    val_metrics["total_file_rows"] = total_val
    results_summary.append(val_metrics)
    del val_ei, val_ea, val_y, val_loader
    gc.collect()
    print(
        f"         + Hoàn tất Validation ({val_metrics['rows']:,} cạnh) trong {val_metrics['eval_time_sec']}s: "
        f"F1={val_metrics['f1']:.4f}, Recall={val_metrics['recall']:.4f}, PR-AUC={val_metrics['pr_auc']:.4f}"
    )

    # --- 3.2. Test_seen ---
    print("      -> (2/3) Đang đánh giá trên tập Test_seen ('data/processed/edges_test_seen.parquet')...")
    ts_ei, ts_ea, ts_y, total_ts = load_edge_tensors(test_seen_parquet, max_edges=eval_seen_max_edges)
    ts_loader = build_eval_loader(train_graph, ts_ei, ts_y)
    ts_metrics, ts_y_true, ts_y_prob = evaluate_model(model, ts_loader, ts_ea, device)
    ts_metrics["split"] = "Test_seen"
    ts_metrics["total_file_rows"] = total_ts
    results_summary.append(ts_metrics)

    # Giữ lại xác suất dự đoán của các mẫu Benign trong Test_seen để tính PR-AUC ghép 1:1 với Unseen Attack
    benign_probs_ref = ts_y_prob[ts_y_true == 0]
    del ts_ei, ts_ea, ts_y, ts_loader, ts_y_true, ts_y_prob
    gc.collect()
    print(
        f"         + Hoàn tất Test_seen ({ts_metrics['rows']:,} cạnh) trong {ts_metrics['eval_time_sec']}s: "
        f"F1={ts_metrics['f1']:.4f}, Recall={ts_metrics['recall']:.4f}, PR-AUC={ts_metrics['pr_auc']:.4f}"
    )

    # --- 3.3. Test_unseen (Unseen: TOÀN BỘ 259,458 cạnh Bot & Infilteration) ---
    print("      -> (3/3) Đang đánh giá trên TOÀN BỘ tập Test_unseen ('data/processed/edges_test_unseen.parquet')...")
    tu_ei, tu_ea, tu_y, total_tu = load_edge_tensors(test_unseen_parquet, max_edges=None)
    tu_loader = build_eval_loader(train_graph, tu_ei, tu_y)
    tu_metrics, tu_y_true, tu_y_prob = evaluate_model(model, tu_loader, tu_ea, device)
    tu_metrics["split"] = "Test_unseen (Unseen Attack)"
    tu_metrics["total_file_rows"] = total_tu

    # Tính thêm PR-AUC, F1, Precision khi ghép cân bằng 1:1 với mẫu Benign từ Test_seen (như đã làm với XGBoost)
    n_unseen = len(tu_y_true)
    ben_sample = benign_probs_ref[:n_unseen]
    y_comb = np.concatenate([tu_y_true, np.zeros(len(ben_sample), dtype=np.int8)])
    prob_comb = np.concatenate([tu_y_prob, ben_sample])
    pred_comb = (prob_comb >= 0.5).astype(np.int8)

    tu_metrics["combined_with_benign_precision"] = float(precision_score(y_comb, pred_comb, zero_division=0))
    tu_metrics["combined_with_benign_f1"] = float(f1_score(y_comb, pred_comb, zero_division=0))
    tu_metrics["combined_with_benign_pr_auc"] = float(average_precision_score(y_comb, prob_comb))

    # Phân tích chi tiết theo từng loại tấn công Unseen (Bot vs Infilteration)
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
    tu_metrics["unseen_attack_breakdown"] = per_attack_recall
    results_summary.append(tu_metrics)

    del tu_ei, tu_ea, tu_y, tu_loader, tu_y_true, tu_y_prob, benign_probs_ref
    gc.collect()

    # Lưu kết quả ra file JSON
    os.makedirs(os.path.join("results", "metrics"), exist_ok=True)
    metrics_out_path = os.path.join("results", "metrics", "gnn_g0_metrics.json")
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # 4. BÁO CÁO BẢNG SO SÁNH 4 CHỈ SỐ & ĐỐI CHIẾU VỚI XGBOOST (B0)
    # =========================================================================
    print("\n" + "=" * 88)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH GNN BASELINE (G0 - GCN)")
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

    xgb_recall = 0.0216
    xgb_f1 = 0.0422
    xgb_comb_prauc = 0.6736
    gcn_recall = tu_metrics["recall"]
    diff_recall = (gcn_recall - xgb_recall) * 100

    print("\n[SO SÁNH TRỰC TIẾP TRÊN TẬP TEST_UNSEEN (UNSEEN: BOT & INFILTERATION)]")
    print(f"- Baseline B0 (XGBoost phi đồ thị) : Recall = {xgb_recall*100:6.2f}% | F1 = {xgb_f1*100:6.2f}% | PR-AUC (ghép Benign) = {xgb_comb_prauc:.4f}")
    print(
        f"- Baseline G0 (GCN đồ thị)         : Recall = {gcn_recall*100:6.2f}% | F1 = {tu_metrics['f1']*100:6.2f}% | "
        f"PR-AUC (ghép Benign) = {tu_metrics['combined_with_benign_pr_auc']:.4f}"
    )
    print(f"- Chênh lệch Recall (G0 so với B0) : {diff_recall:+.2f}% ({tu_metrics['TP']:,}/{tu_metrics['rows']:,} mẫu Unseen được phát hiện)")
    print("- Chi tiết theo từng họ tấn công Unseen (G0 - GCN):")
    for atk, info in tu_metrics["unseen_attack_breakdown"].items():
        print(
            f"   + {atk:<10}: {info['detected_tp']:>6,}/{info['count']:<6,} phát hiện được "
            f"| Recall = {info['recall']*100:>6.2f}% | Mean Prob = {info['mean_prob']:.4f}"
        )
    print("=" * 88)
    print(f"Tổng thời gian thực thi toàn bộ script: {time.time() - start_time:.1f} giây")


if __name__ == "__main__":
    main()
