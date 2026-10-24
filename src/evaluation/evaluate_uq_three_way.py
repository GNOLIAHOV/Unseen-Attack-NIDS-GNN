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
EVAL_BATCH_SIZE = 8192
NUM_NEIGHBORS = [10, 10]
MC_SAMPLES_T = 20
STD_THRESHOLD = 0.15  # Ngưỡng độ lệch chuẩn Epistemic Std cho quy tắc 3 nhánh
NON_FEATURE_COLS = ["src_node_id", "dst_node_id", "label", "Attack", "Dataset"]


class GNNInductiveGATv2(nn.Module):
    """
    Kiến trúc G1_v2 (Inductive Edge-Aware GATv2 + LayerNorm) tích hợp Monte Carlo (MC) Dropout
    cho phép trích xuất đồng thời Mean Probability và Epistemic Uncertainty (Std) qua T=20 lần lấy mẫu.
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
        edge_label_index: torch.Tensor,
        target_edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.gelu(self.node_proj(x))
        e_sub = self.edge_norm(edge_attr)
        e_target = self.edge_norm(target_edge_attr)

        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        bi_edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
        bi_edge_attr = torch.cat([e_sub, e_sub], dim=0)
        return h, bi_edge_index, bi_edge_attr, edge_label_index, e_target

    def forward_single_mc(
        self,
        h: torch.Tensor,
        bi_edge_index: torch.Tensor,
        bi_edge_attr: torch.Tensor,
        edge_label_index: torch.Tensor,
        e_target: torch.Tensor,
    ) -> torch.Tensor:
        h1 = self.gat1(h, bi_edge_index, edge_attr=bi_edge_attr)
        h1 = F.gelu(self.norm1(h1))
        h1 = self.mc_node_drop(h1)

        h2 = self.gat2(h1, bi_edge_index, edge_attr=bi_edge_attr)
        h2 = F.gelu(self.norm2(h2 + h1))
        h2 = self.mc_node_drop(h2)

        src_idx, dst_idx = edge_label_index[0], edge_label_index[1]
        node_src = h2[src_idx]
        node_dst = h2[dst_idx]

        edge_repr = torch.cat([node_src, node_dst, e_target], dim=-1)
        logits = self.classifier(edge_repr).squeeze(-1)
        return torch.sigmoid(logits)


def load_graph_split(
    parquet_path: str,
    static_node_x: torch.Tensor,
    max_edges: int | None = None,
) -> tuple[Data, int]:
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

    data = Data(
        x=static_node_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )
    return data, total_edges


@torch.no_grad()
def evaluate_three_way_uq(
    model: GNNInductiveGATv2,
    loader: LinkNeighborLoader,
    full_target_edge_attr: torch.Tensor,
    device: torch.device,
    T: int = MC_SAMPLES_T,
    std_threshold: float = STD_THRESHOLD,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Triển khai cơ chế phân loại 3 nhánh (Three-way Decision) dựa trên MC Dropout (T=20):
      - Nhánh 1 (Trusted Benign)    : Mean Prob < 0.5  VÀ Std <= 0.15  -> Predict = 0
      - Nhánh 2 (Known Attack)      : Mean Prob >= 0.5 VÀ Std <= 0.15  -> Predict = 1
      - Nhánh 3 (Zero-day/OOD Alert): Std > 0.15 (Bất định cao)        -> Predict = 1
    """
    t0 = time.time()

    det_probs_list = []
    mc_mean_list = []
    mc_std_list = []
    labels_list = []

    for batch in loader:
        batch = batch.to(device)
        target_edge_attr = full_target_edge_attr[batch.input_id.cpu()].to(device)

        # 1) Dự đoán Deterministic trước UQ (model.eval())
        model.eval()
        h, bi_ei, bi_ea, eli, e_target = model.encode_subgraph(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            edge_label_index=batch.edge_label_index,
            target_edge_attr=target_edge_attr,
        )
        det_prob = model.forward_single_mc(h, bi_ei, bi_ea, eli, e_target)

        # 2) Thực hiện T=20 lần forward pass với MC Dropout (model.train() + torch.no_grad())
        model.train()
        mc_preds = []
        for _ in range(T):
            prob_t = model.forward_single_mc(h, bi_ei, bi_ea, eli, e_target)
            mc_preds.append(prob_t)

        mc_stack = torch.stack(mc_preds, dim=0)  # [T, batch_size]
        mean_prob = mc_stack.mean(dim=0)
        epistemic_std = mc_stack.std(dim=0, unbiased=False)

        det_probs_list.append(det_prob.cpu().numpy().astype(np.float32))
        mc_mean_list.append(mean_prob.cpu().numpy().astype(np.float32))
        mc_std_list.append(epistemic_std.cpu().numpy().astype(np.float32))
        labels_list.append(batch.edge_label.cpu().numpy().astype(np.int8))

    y_true = np.concatenate(labels_list)
    det_probs = np.concatenate(det_probs_list)
    mean_probs = np.concatenate(mc_mean_list)
    std_vals = np.concatenate(mc_std_list)

    del det_probs_list, mc_mean_list, mc_std_list, labels_list
    gc.collect()

    # --- A. Kết quả TRƯỚC UQ (Deterministic: p >= 0.5) ---
    pred_before = (det_probs >= 0.5).astype(np.int8)
    before_metrics = {
        "precision": float(precision_score(y_true, pred_before, zero_division=0)),
        "recall": float(recall_score(y_true, pred_before, zero_division=0)),
        "f1": float(f1_score(y_true, pred_before, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, det_probs)),
        "TP": int(((y_true == 1) & (pred_before == 1)).sum()),
        "FP": int(((y_true == 0) & (pred_before == 1)).sum()),
        "TN": int(((y_true == 0) & (pred_before == 0)).sum()),
        "FN": int(((y_true == 1) & (pred_before == 0)).sum()),
    }

    # --- B. Phân chia 3 nhánh (Three-way Decision) ---
    branch1_mask = (mean_probs < 0.5) & (std_vals <= std_threshold)   # Nhánh 1: Trusted Benign (0)
    branch2_mask = (mean_probs >= 0.5) & (std_vals <= std_threshold)  # Nhánh 2: Known Attack (1)
    branch3_mask = std_vals > std_threshold                           # Nhánh 3: Zero-day / OOD Alert (1)

    # Đối với hệ thống IDS/Firewall, cả Nhánh 2 ("Known Attack") và Nhánh 3 ("Zero-day Alert")
    # đều được đánh chặn (Predict = 1):
    pred_three_way = (branch2_mask | branch3_mask).astype(np.int8)

    # Điểm hợp nhất UQ (Combined Risk Score) cho PR-AUC
    combined_risk_score = np.maximum(mean_probs, np.clip(std_vals / 0.30, 0.0, 1.0))

    three_way_metrics = {
        "precision": float(precision_score(y_true, pred_three_way, zero_division=0)),
        "recall": float(recall_score(y_true, pred_three_way, zero_division=0)),
        "f1": float(f1_score(y_true, pred_three_way, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, combined_risk_score)),
        "TP": int(((y_true == 1) & (pred_three_way == 1)).sum()),
        "FP": int(((y_true == 0) & (pred_three_way == 1)).sum()),
        "TN": int(((y_true == 0) & (pred_three_way == 0)).sum()),
        "FN": int(((y_true == 1) & (pred_three_way == 0)).sum()),
        "branch1_trusted_benign_count": int(branch1_mask.sum()),
        "branch2_known_attack_count": int(branch2_mask.sum()),
        "branch3_zeroday_alert_count": int(branch3_mask.sum()),
        "tp_from_branch2_known": int(((y_true == 1) & branch2_mask).sum()),
        "tp_from_branch3_zeroday_uq": int(((y_true == 1) & branch3_mask).sum()),
        "mean_prob_avg": float(np.mean(mean_probs)),
        "epistemic_std_avg": float(np.mean(std_vals)),
    }

    # --- C. Thống kê trên tập các quyết định tự tin (Selective / Confident Zone: Nhánh 1 & Nhánh 2) ---
    # Khi Nhánh 3 được chuyển tiếp sang bộ phân tích chuyên sâu (SOC / Deep Packet Inspection),
    # luồng tự động hoàn toàn (Nhánh 1 vs Nhánh 2) đạt độ chính xác rất cao:
    confident_mask = ~branch3_mask
    if int(confident_mask.sum()) > 0 and int((y_true[confident_mask] == 1).sum()) > 0:
        y_conf = y_true[confident_mask]
        p_conf = branch2_mask[confident_mask].astype(np.int8)
        three_way_metrics["confident_zone_coverage"] = float(confident_mask.mean())
        three_way_metrics["confident_zone_precision"] = float(precision_score(y_conf, p_conf, zero_division=0))
        three_way_metrics["confident_zone_recall"] = float(recall_score(y_conf, p_conf, zero_division=0))
        three_way_metrics["confident_zone_f1"] = float(f1_score(y_conf, p_conf, zero_division=0))
        three_way_metrics["confident_zone_fp"] = int(((y_conf == 0) & (p_conf == 1)).sum())

    res = {
        "rows": int(len(y_true)),
        "eval_time_sec": round(time.time() - t0, 2),
        "before_uq": before_metrics,
        "three_way_uq": three_way_metrics,
    }
    return res, y_true, mean_probs, std_vals


def main():
    start_time = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join("models", "gnns", "gatv2_g1_v2_model.pt")
    test_seen_path = os.path.join("data", "processed", "edges_test_seen.parquet")
    test_unseen_path = os.path.join("data", "processed", "edges_test_unseen.parquet")
    out_metrics_path = os.path.join("results", "metrics", "uq_three_way_metrics.json")

    print("=" * 94)
    print("PHÁT HIỆN ZERO-DAY DỰA TRÊN ĐỊNH LƯỢNG BẤT ĐỊNH BA NHÁNH (THREE-WAY DECISION UQ - T=20)")
    print("=" * 94)
    print(f"- Mô hình nạp từ             : {model_path}")
    print(f"- Số lần lấy mẫu MC (T)      : {MC_SAMPLES_T} forward passes / batch")
    print(f"- Quy tắc 3 nhánh (Std={STD_THRESHOLD}) :")
    print(f"   + Nhánh 1 (Trusted Benign)    : Mean Prob < 0.5  VÀ Std <= {STD_THRESHOLD} -> Predict = 0")
    print(f"   + Nhánh 2 (Known Attack)      : Mean Prob >= 0.5 VÀ Std <= {STD_THRESHOLD} -> Predict = 1")
    print(f"   + Nhánh 3 (Zero-day/OOD Alert): Std > {STD_THRESHOLD} (Bất định cao)       -> Predict = 1\n")

    # 1. Load mô hình GATv2
    print("[1/3] Đang nạp mô hình GATv2 từ 'models/gnns/gatv2_g1_v2_model.pt'...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = GNNInductiveGATv2(
        node_init_dim=checkpoint.get("node_init_dim", NODE_INIT_DIM),
        hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
        heads=checkpoint.get("heads", NUM_HEADS),
        edge_attr_dim=checkpoint.get("edge_attr_dim", EDGE_ATTR_DIM),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("      -> Nạp mô hình thành công!\n")

    static_node_x = torch.ones((NUM_TOTAL_NODES, NODE_INIT_DIM), dtype=torch.float32)
    test_seen_eval_edges = 200_000

    # 2. Đánh giá trên tập Test_seen
    print(f"[2/3] Đang chạy Three-way UQ (T={MC_SAMPLES_T}) trên tập Test_seen ('{test_seen_path}')...")
    ts_data, total_ts = load_graph_split(test_seen_path, static_node_x, max_edges=test_seen_eval_edges)
    ts_loader = LinkNeighborLoader(
        data=ts_data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=EVAL_BATCH_SIZE,
        edge_label_index=ts_data.edge_index,
        edge_label=ts_data.y,
        shuffle=False,
    )
    ts_res, _, _, _ = evaluate_three_way_uq(
        model,
        ts_loader,
        ts_data.edge_attr,
        device,
        T=MC_SAMPLES_T,
        std_threshold=STD_THRESHOLD,
    )
    ts_res["split"] = "Test_seen"
    ts_res["total_file_rows"] = total_ts
    del ts_loader, ts_data
    gc.collect()
    print(f"      -> Hoàn tất Test_seen ({ts_res['rows']:,} cạnh) trong {ts_res['eval_time_sec']}s.\n")

    # 3. Đánh giá trên TOÀN BỘ tập Test_unseen (Zero-day: 53,861 cạnh)
    print(f"[3/3] Đang chạy Three-way UQ (T={MC_SAMPLES_T}) trên TOÀN BỘ tập Test_unseen ('{test_unseen_path}')...")
    tu_data, total_tu = load_graph_split(test_unseen_path, static_node_x, max_edges=None)
    tu_loader = LinkNeighborLoader(
        data=tu_data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=EVAL_BATCH_SIZE,
        edge_label_index=tu_data.edge_index,
        edge_label=tu_data.y,
        shuffle=False,
    )
    tu_res, tu_y_true, tu_mean, tu_std = evaluate_three_way_uq(
        model,
        tu_loader,
        tu_data.edge_attr,
        device,
        T=MC_SAMPLES_T,
        std_threshold=STD_THRESHOLD,
    )
    tu_res["split"] = "Test_unseen (Zero-day)"
    tu_res["total_file_rows"] = total_tu

    # Thống kê chi tiết theo từng họ tấn công Zero-day (Exploits vs Fuzzers)
    attacks = (
        pl.scan_parquet(test_unseen_path)
        .select("Attack")
        .collect(engine="streaming")["Attack"]
        .to_list()
    )
    attack_arr = np.array(attacks)
    b2_mask = (tu_mean >= 0.5) & (tu_std <= STD_THRESHOLD)
    b3_mask = tu_std > STD_THRESHOLD
    pred_tu = (b2_mask | b3_mask).astype(np.int8)

    per_attack_stats = {}
    for atk in np.unique(attack_arr):
        m = attack_arr == atk
        per_attack_stats[str(atk)] = {
            "total": int(m.sum()),
            "detected_total": int(pred_tu[m].sum()),
            "recall_total": float(pred_tu[m].mean()),
            "caught_by_branch2_known": int(b2_mask[m].sum()),
            "caught_by_branch3_uq_alert": int(b3_mask[m].sum()),
            "mean_std": float(tu_std[m].mean()),
        }
    tu_res["zero_day_breakdown"] = per_attack_stats

    del tu_loader, tu_data, tu_y_true, tu_mean, tu_std
    gc.collect()
    print(f"      -> Hoàn tất Test_unseen ({tu_res['rows']:,} cạnh) trong {tu_res['eval_time_sec']}s.\n")

    # Lưu kết quả ra JSON
    os.makedirs(os.path.dirname(out_metrics_path), exist_ok=True)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump([ts_res, tu_res], f, indent=2, ensure_ascii=False)

    # =========================================================================
    # IN BẢNG BÁO CÁO TỔNG HỢP
    # =========================================================================
    print("=" * 94)
    print("BẢNG KẾT QUẢ PHÂN LOẠI BA NHÁNH (THREE-WAY DECISION UQ - MC DROPOUT T=20)")
    print("=" * 94)
    print(
        f"{'Tập dữ liệu':<22} | {'Cơ chế':<22} | {'Precision':>10} | {'Recall':>10} | {'F1-score':>10} | {'FP (Báo giả)':>12}"
    )
    print("-" * 94)

    for res in [ts_res, tu_res]:
        sp = res["split"]
        b = res["before_uq"]
        tw = res["three_way_uq"]
        print(
            f"{sp:<22} | {'Trước UQ (p >= 0.5)':<22} | {b['precision']:>10.4f} | "
            f"{b['recall']:>10.4f} | {b['f1']:>10.4f} | {b['FP']:>12,}"
        )
        print(
            f"{sp:<22} | {'Three-way UQ (All=1)':<22} | {tw['precision']:>10.4f} | "
            f"{tw['recall']:>10.4f} | {tw['f1']:>10.4f} | {tw['FP']:>12,}"
        )
        if "confident_zone_precision" in tw:
            print(
                f"{sp:<22} | {'Nhánh 1&2 (Tự tin)':<22} | {tw['confident_zone_precision']:>10.4f} | "
                f"{tw['confident_zone_recall']:>10.4f} | {tw['confident_zone_f1']:>10.4f} | {tw['confident_zone_fp']:>12,}"
            )
        print("-" * 94)

    # Phân tích chi tiết từng nhánh
    print("\n[THỐNG KÊ CHI TIẾT PHÂN BỐ 3 NHÁNH TRÊN TỪNG TẬP DỮ LIỆU]")
    for res in [ts_res, tu_res]:
        sp = res["split"]
        tw = res["three_way_uq"]
        total = res["rows"]
        b1_c = tw["branch1_trusted_benign_count"]
        b2_c = tw["branch2_known_attack_count"]
        b3_c = tw["branch3_zeroday_alert_count"]
        print(f"* {sp} (Tổng: {total:,} luồng):")
        print(f"  - Nhánh 1 (Trusted Benign  : Mean < 0.5 & Std <= 0.15) -> Predict 0 : {b1_c:>8,} luồng ({b1_c/total*100:5.2f}%)")
        print(f"  - Nhánh 2 (Known Attack    : Mean >= 0.5 & Std <= 0.15)-> Predict 1 : {b2_c:>8,} luồng ({b2_c/total*100:5.2f}%)")
        print(f"  - Nhánh 3 (Zero-day Alert  : Std > 0.15 - Bất định cao)-> Predict 1 : {b3_c:>8,} luồng ({b3_c/total*100:5.2f}%)")

    tu_tw = tu_res["three_way_uq"]
    print("\n[TRỌNG TÂM: KHẢ NĂNG BẮT TẤN CÔNG ZERO-DAY NHỜ TÍN HIỆU BẤT ĐỊNH (NHÁNH 3)]")
    print(f"- Tổng số mẫu Zero-day trong Test_unseen             : {tu_res['rows']:,} mẫu")
    print(
        f"- Tổng số mẫu Zero-day bắt thành công (Nhánh 2 + 3)  : {tu_tw['TP']:,} / {tu_res['rows']:,} mẫu "
        f"(Recall đạt {tu_tw['recall']*100:.2f}%, F1 đạt {tu_tw['f1']*100:.2f}%!)"
    )
    print(
        f"   + Bắt được bởi Nhánh 2 (Known Attack, Std <= 0.15): {tu_tw['tp_from_branch2_known']:,} mẫu "
        f"({tu_tw['tp_from_branch2_known']/tu_res['rows']*100:.2f}%)"
    )
    print(
        f"   + Bắt được bởi Nhánh 3 (Zero-day UQ, Std > 0.15)  : {tu_tw['tp_from_branch3_zeroday_uq']:,} mẫu "
        f"({tu_tw['tp_from_branch3_zeroday_uq']/tu_res['rows']*100:.2f}% tổng số mẫu Zero-day!)"
    )
    print("- Chi tiết theo từng họ tấn công Zero-day:")
    for atk, st in tu_res["zero_day_breakdown"].items():
        print(
            f"   + {atk:<10}: Tổng bắt được {st['detected_total']:>6,}/{st['total']:<6,} (Recall = {st['recall_total']*100:5.2f}%) "
            f"| Nhánh 3 (Bất định cao) bắt = {st['caught_by_branch3_uq_alert']:>6,} mẫu | Mean Std = {st['mean_std']:.4f}"
        )
    print("=" * 94)
    print(f"Tổng thời gian thực thi: {time.time() - start_time:.1f} giây (Đã lưu '{out_metrics_path}')")


if __name__ == "__main__":
    main()
