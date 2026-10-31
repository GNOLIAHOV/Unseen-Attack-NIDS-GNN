import gc
import json
import os
import sys
import time
import warnings
import numpy as np
import polars as pl
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EVAL_BATCH_SIZE = 8192
NUM_NEIGHBORS = [10, 10]
MC_SAMPLES_T = 20
OSR_THRESHOLD = 0.15


def load_graph_split(parquet_path: str, max_edges: int | None = None) -> tuple[Data, int]:
    """Đọc file Parquet từ data/processed/ và đóng gói thành đối tượng PyG Data."""
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

    node_x = compute_inductive_node_features(edge_index, num_nodes=NUM_TOTAL_NODES)
    return Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr, y=y), total_edges


def compute_oscr(
    known_true: np.ndarray,
    known_pred_closed: np.ndarray,
    known_osr_score: np.ndarray,
    unknown_osr_score: np.ndarray,
    num_thresholds: int = 200,
) -> float:
    """
    Tính chỉ số chuẩn học thuật OSCR (Open-Set Classification Rate):
    Diện tích dưới đường cong CCR (Correct Classification Rate trên tập Known được giữ lại)
    theo FPR_unknown (Tỷ lệ mẫu Unknown bị lọt vào vùng Known).
    """
    all_scores = np.concatenate([known_osr_score, unknown_osr_score])
    thresholds = np.quantile(all_scores, np.linspace(0.0, 1.0, num_thresholds))
    correct_closed = known_pred_closed == known_true

    ccr_list = []
    fpr_u_list = []
    for tau in thresholds:
        # Mẫu được chấp nhận là Known nếu osr_score <= tau
        ccr = float(np.mean(correct_closed & (known_osr_score <= tau)))
        fpr_u = float(np.mean(unknown_osr_score <= tau))
        ccr_list.append(ccr)
        fpr_u_list.append(fpr_u)

    order = np.argsort(fpr_u_list)
    fpr_sorted = np.array(fpr_u_list)[order]
    ccr_sorted = np.array(ccr_list)[order]
    return float(np.trapezoid(ccr_sorted, fpr_sorted))


@torch.no_grad()
def extract_osr_signals(
    model: GNNInductiveGATv2,
    loader: LinkNeighborLoader,
    full_target_edge_attr: torch.Tensor,
    device: torch.device,
    T: int = MC_SAMPLES_T,
) -> dict[str, np.ndarray]:
    """
    Trích xuất đồng thời các tín hiệu cho bài toán Open-Set Recognition (OSR):
      - det_prob      : Xác suất tất định (T=1)
      - p_topo, p_flow: Xác suất từ nhánh cấu trúc đồ thị (Topo) và nhánh đặc trưng luồng (Flow)
      - discrepancy   : |p_topo - p_flow| (Độ bất đồng giữa cấu trúc mạng và header gói tin)
      - mc_mean       : Xác suất trung bình qua T=20 lượt lấy mẫu MC Dropout
      - mc_std        : Độ lệch chuẩn nhận thức (Epistemic Uncertainty Std)
      - margin_score  : Điểm bất định lề tất định 1 - 2*|det_prob - 0.5|
      - hybrid_osr    : Điểm nhận dạng tập mở lai ghép (0.65 * mc_std + 0.35 * 0.5 * discrepancy)
    """
    det_probs, topo_probs, flow_probs = [], [], []
    mc_means, mc_stds, labels = [], [], []

    for batch in loader:
        batch = batch.to(device)
        target_edge_attr = full_target_edge_attr[batch.input_id.cpu()].to(device)

        model.eval()
        h, bi_ei, bi_ea, eli, e_target = model.encode_subgraph(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            edge_label_index=batch.edge_label_index,
            target_edge_attr=target_edge_attr,
        )
        l_topo, l_flow = model.forward_dual_logits(h, bi_ei, bi_ea, eli, e_target)
        p_t = torch.sigmoid(l_topo)
        p_f = torch.sigmoid(l_flow)
        p_det = torch.sigmoid(0.55 * l_topo + 0.45 * l_flow)

        model.train()
        mc_preds = [model.forward_single_mc(h, bi_ei, bi_ea, eli, e_target) for _ in range(T)]
        mc_stack = torch.stack(mc_preds, dim=0)
        m_prob = mc_stack.mean(dim=0)
        s_prob = mc_stack.std(dim=0, unbiased=False)

        det_probs.append(p_det.cpu().numpy().astype(np.float32))
        topo_probs.append(p_t.cpu().numpy().astype(np.float32))
        flow_probs.append(p_f.cpu().numpy().astype(np.float32))
        mc_means.append(m_prob.cpu().numpy().astype(np.float32))
        mc_stds.append(s_prob.cpu().numpy().astype(np.float32))
        labels.append(batch.edge_label.cpu().numpy().astype(np.int8))

    det_p = np.concatenate(det_probs)
    p_topo = np.concatenate(topo_probs)
    p_flow = np.concatenate(flow_probs)
    mc_m = np.concatenate(mc_means)
    mc_s = np.concatenate(mc_stds)
    y_true = np.concatenate(labels)

    discrepancy = np.abs(p_topo - p_flow)
    margin_score = 1.0 - 2.0 * np.abs(det_p - 0.5)
    # Điểm OSR kết hợp Epistemic Std (MC Dropout) và độ lệch giữa hai nhánh (Dual-Pathway Discrepancy)
    hybrid_osr = np.maximum(mc_s, 0.35 * discrepancy)

    return {
        "y_true": y_true,
        "det_prob": det_p,
        "p_topo": p_topo,
        "p_flow": p_flow,
        "discrepancy": discrepancy,
        "mc_mean": mc_m,
        "mc_std": mc_s,
        "margin_score": margin_score,
        "hybrid_osr": hybrid_osr,
    }


def main():
    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join("models", "gnns", "gatv2_g1_v2_model.pt")
    test_seen_path = os.path.join("data", "processed", "edges_test_seen.parquet")
    test_unseen_path = os.path.join("data", "processed", "edges_test_unseen.parquet")
    out_metrics_path = os.path.join("results", "metrics", "osr_open_set_metrics.json")

    print("=" * 96)
    print("ĐÁNH GIÁ MỤC 15.1: NHẬN DẠNG TẬP MỞ (OPEN-SET RECOGNITION - OSR) TRÊN ĐỒ THỊ NETFLOW")
    print("=" * 96)
    print(f"- Mô hình nạp từ             : {model_path}")
    print(f"- Không gian nhãn Tập Mở     : {{0: Known Benign, 1: Known Attack, 2: Unknown/Unseen Attack}}")
    print(f"- Số lần lấy mẫu MC (T)      : {MC_SAMPLES_T} forward passes / batch")
    print(f"- Ngưỡng phát hiện Tập Mở    : tau = {OSR_THRESHOLD}\n")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = GNNInductiveGATv2(
        node_init_dim=checkpoint.get("node_init_dim", NODE_INIT_DIM),
        hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
        heads=checkpoint.get("heads", NUM_HEADS),
        edge_attr_dim=checkpoint.get("edge_attr_dim", EDGE_ATTR_DIM),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("[1/3] Đang trích xuất tín hiệu OSR trên tập In-Distribution (Test_seen: 200,000 luồng)...")
    ts_data, total_ts = load_graph_split(test_seen_path, max_edges=200_000)
    ts_loader = LinkNeighborLoader(
        data=ts_data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=EVAL_BATCH_SIZE,
        edge_label_index=ts_data.edge_index,
        edge_label=ts_data.y,
        shuffle=False,
    )
    sig_seen = extract_osr_signals(model, ts_loader, ts_data.edge_attr, device, T=MC_SAMPLES_T)
    del ts_loader, ts_data
    gc.collect()

    print("[2/3] Đang trích xuất tín hiệu OSR trên tập Open-Set Unknown (Test_unseen: 259,458 luồng Bot & Infilteration)...")
    tu_data, total_tu = load_graph_split(test_unseen_path, max_edges=None)
    tu_loader = LinkNeighborLoader(
        data=tu_data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=EVAL_BATCH_SIZE,
        edge_label_index=tu_data.edge_index,
        edge_label=tu_data.y,
        shuffle=False,
    )
    sig_unseen = extract_osr_signals(model, tu_loader, tu_data.edge_attr, device, T=MC_SAMPLES_T)
    del tu_loader, tu_data
    gc.collect()

    print("[3/3] Đang tính toán các chỉ số chuẩn Open-Set Recognition (AUROC, AUPR-Out, OSCR, 3x3 Confusion Matrix)...")
    n_seen = len(sig_seen["y_true"])
    n_unseen = len(sig_unseen["y_true"])

    # Nhãn nhị phân cho bài toán phát hiện Unknown (0 = Known In-Distribution, 1 = Unknown Open-Set)
    y_ood = np.concatenate([np.zeros(n_seen, dtype=np.int8), np.ones(n_unseen, dtype=np.int8)])
    known_true = sig_seen["y_true"]
    known_pred_t1 = (sig_seen["det_prob"] >= 0.5).astype(np.int8)
    known_pred_mc = (sig_seen["mc_mean"] >= 0.5).astype(np.int8)

    methods_cfg = [
        (
            "1. Closed-Set Sigmoid Margin (T=1)",
            sig_seen["margin_score"],
            sig_unseen["margin_score"],
            known_pred_t1,
            0.50,
        ),
        (
            "2. Dual-Pathway Energy Discrepancy (T=1)",
            0.5 * sig_seen["discrepancy"],
            0.5 * sig_unseen["discrepancy"],
            known_pred_t1,
            0.15,
        ),
        (
            "3. Three-way Epistemic MC Dropout (T=20)",
            sig_seen["mc_std"],
            sig_unseen["mc_std"],
            known_pred_mc,
            OSR_THRESHOLD,
        ),
        (
            "4. Hybrid Epistemic + Dual-Pathway OSR (T=20 - Proposed)",
            sig_seen["hybrid_osr"],
            sig_unseen["hybrid_osr"],
            known_pred_mc,
            OSR_THRESHOLD,
        ),
    ]

    method_comparisons = []
    for name, s_k, s_u, k_pred, thr in methods_cfg:
        s_all = np.concatenate([s_k, s_u])
        auroc = float(roc_auc_score(y_ood, s_all))
        aupr_out = float(average_precision_score(y_ood, s_all))
        oscr = compute_oscr(known_true, k_pred, s_k, s_u)

        # Dự đoán 3 lớp Open-Set: {0: Known Benign, 1: Known Attack, 2: Unknown/Unseen Attack}
        pred_seen_3c = np.where(s_k > thr, 2, k_pred)
        pred_unseen_3c = np.where(
            s_u > thr,
            2,
            (sig_unseen["mc_mean"] >= 0.5).astype(np.int8),
        )

        # Tỷ lệ phát hiện đúng mẫu Unknown (Open-Set Recall = bắt vào lớp 2, và Total Threat Recall = lớp 1 + lớp 2)
        unk_flagged_as_2 = float(np.mean(pred_unseen_3c == 2))
        unk_total_intercepted = float(np.mean(pred_unseen_3c >= 1))
        known_ccr = float(np.mean(pred_seen_3c == known_true))

        method_comparisons.append(
            {
                "method": name,
                "threshold": thr,
                "auroc_open_set": round(auroc, 4),
                "aupr_unknown": round(aupr_out, 4),
                "oscr_score": round(oscr, 4),
                "known_class_ccr": round(known_ccr, 4),
                "unknown_flagged_as_open_set_rate": round(unk_flagged_as_2, 4),
                "unknown_total_threat_recall": round(unk_total_intercepted, 4),
                "mean_score_known": round(float(np.mean(s_k)), 4),
                "mean_score_unknown": round(float(np.mean(s_u)), 4),
            }
        )

    # Phân tích ma trận nhầm lẫn 3x3 (3-Class Open-Set Confusion Matrix) cho cơ chế Three-way / Hybrid OSR
    s_k_prop = sig_seen["hybrid_osr"]
    s_u_prop = sig_unseen["hybrid_osr"]

    y_true_3class = np.concatenate([known_true, np.full(n_unseen, 2, dtype=np.int8)])
    pred_seen_prop = np.where(s_k_prop > OSR_THRESHOLD, 2, known_pred_mc)
    pred_unseen_prop = np.where(
        s_u_prop > OSR_THRESHOLD,
        2,
        (sig_unseen["mc_mean"] >= 0.5).astype(np.int8),
    )
    y_pred_3class = np.concatenate([pred_seen_prop, pred_unseen_prop])

    cm_3x3 = confusion_matrix(y_true_3class, y_pred_3class, labels=[0, 1, 2]).tolist()
    macro_f1_3class = float(f1_score(y_true_3class, y_pred_3class, average="macro"))

    # Nếu quy về bài toán Phòng thủ Mạng (Benign = 0 vs Threat {Known Attack U Unknown Attack} = 1):
    y_true_threat = (y_true_3class >= 1).astype(np.int8)
    y_pred_threat = (y_pred_3class >= 1).astype(np.int8)
    threat_precision = float(precision_score(y_true_threat, y_pred_threat, zero_division=0))
    threat_recall = float(recall_score(y_true_threat, y_pred_threat, zero_division=0))
    threat_f1 = float(f1_score(y_true_threat, y_pred_threat, zero_division=0))

    # Chi tiết theo từng họ tấn công Unknown (Bot vs Infilteration)
    attacks = (
        pl.scan_parquet(test_unseen_path)
        .select("Attack")
        .collect(engine="streaming")["Attack"]
        .to_numpy()
    )
    family_osr = {}
    for atk in np.unique(attacks):
        m = attacks == atk
        p_3c = pred_unseen_prop[m]
        family_osr[str(atk)] = {
            "total_flows": int(m.sum()),
            "predicted_class_0_benign_missed": int((p_3c == 0).sum()),
            "predicted_class_1_known_attack": int((p_3c == 1).sum()),
            "predicted_class_2_open_set_unknown": int((p_3c == 2).sum()),
            "open_set_unknown_rate": round(float(np.mean(p_3c == 2)), 4),
            "total_intercept_recall": round(float(np.mean(p_3c >= 1)), 4),
            "mean_epistemic_std": round(float(np.mean(sig_unseen["mc_std"][m])), 4),
            "mean_pathway_discrepancy": round(float(np.mean(sig_unseen["discrepancy"][m])), 4),
            "mean_hybrid_osr_score": round(float(np.mean(s_u_prop[m])), 4),
        }

    summary = {
        "task": "15.1 Open-Set Recognition (OSR) on Host-Flow NetFlow Graph",
        "total_evaluated_flows": int(n_seen + n_unseen),
        "known_in_distribution_flows": int(n_seen),
        "unknown_open_set_flows": int(n_unseen),
        "unknown_attack_families": ["Bot", "Infilteration"],
        "eval_time_sec": round(time.time() - t0, 2),
        "osr_method_comparison": method_comparisons,
        "proposed_hybrid_osr_summary": {
            "threshold": OSR_THRESHOLD,
            "auroc_open_set": method_comparisons[-1]["auroc_open_set"],
            "aupr_unknown": method_comparisons[-1]["aupr_unknown"],
            "oscr_score": method_comparisons[-1]["oscr_score"],
            "macro_f1_3class": round(macro_f1_3class, 4),
            "overall_threat_precision": round(threat_precision, 4),
            "overall_threat_recall": round(threat_recall, 4),
            "overall_threat_f1": round(threat_f1, 4),
            "confusion_matrix_3x3": {
                "labels": [
                    "0: Known Benign",
                    "1: Known Attack",
                    "2: Unknown/Unseen Attack (Bot & Infilteration)",
                ],
                "matrix": cm_3x3,
            },
            "unknown_family_breakdown": family_osr,
        },
    }

    os.makedirs(os.path.dirname(out_metrics_path), exist_ok=True)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 96)
    print("BẢNG SO SÁNH CÁC PHƯƠNG PHÁP NHẬN DẠNG TẬP MỞ (OPEN-SET RECOGNITION BENCHMARK)")
    print("=" * 96)
    print(
        f"{'Phương pháp OSR':<50} | {'AUROC':>7} | {'AUPR-Out':>8} | {'OSCR':>7} | {'Unknown Recall':>14}"
    )
    print("-" * 96)
    for m in method_comparisons:
        print(
            f"{m['method']:<50} | {m['auroc_open_set']:>7.4f} | "
            f"{m['aupr_unknown']:>8.4f} | {m['oscr_score']:>7.4f} | "
            f"{m['unknown_total_threat_recall']*100:>13.2f}%"
        )
    print("-" * 96)
    print(f"- Ma trận nhầm lẫn Tập Mở 3 lớp (0: Benign, 1: Known Attack, 2: Unknown Attack): {cm_3x3}")
    print(
        f"- Hiệu năng Phòng thủ Tổng thể trên dòng chảy Tập Mở ({n_seen + n_unseen:,} luồng): "
        f"Precision = {threat_precision*100:.2f}% | Recall = {threat_recall*100:.2f}% | F1 = {threat_f1*100:.2f}%"
    )
    print(f"- Đã lưu kết quả tại: '{out_metrics_path}' (Thời gian: {summary['eval_time_sec']}s)")
    print("=" * 96)


if __name__ == "__main__":
    main()
