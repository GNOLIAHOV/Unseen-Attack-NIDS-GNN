import gc
import json
import os
import sys
import time
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

# Đảm bảo console trên Windows hiển thị đúng tiếng Việt (UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


NON_FEATURE_COLS = ["src_node_id", "dst_node_id", "label", "Attack", "Dataset"]


def get_feature_cols(parquet_path: str) -> list[str]:
    """Lấy danh sách 41 cột đặc trưng (loại bỏ src_node_id, dst_node_id, label, Attack, Dataset)."""
    schema_cols = pl.scan_parquet(parquet_path).collect_schema().names()
    return [c for c in schema_cols if c not in NON_FEATURE_COLS]


def load_xy_polars(
    parquet_path: str,
    feature_cols: list[str],
    max_rows: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Đọc file Parquet bằng Polars, tách Features (X) ở dạng float32 và Target (y = label) ở dạng int8.
    Hỗ trợ lấy mẫu ngẫu nhiên có phân tầng/xáo trộn nếu cần giới hạn RAM trên tập quá lớn.
    """
    lf = pl.scan_parquet(parquet_path).select(feature_cols + ["label"])
    total_rows = pl.scan_parquet(parquet_path).select(pl.len()).collect(engine="streaming").item()

    if max_rows is not None and total_rows > max_rows:
        print(
            f"      [*] File '{parquet_path}' có {total_rows:,} dòng. "
            f"Đang nạp {max_rows:,} dòng đầu tiên (tập đã được shuffle sẵn với seed={seed}) để tránh tràn RAM..."
        )
        df = lf.head(max_rows).collect(engine="streaming")
    else:
        df = lf.collect(engine="streaming")

    y = df["label"].to_numpy().astype(np.int8, copy=False)
    df_x = df.drop("label").with_columns([pl.col(c).cast(pl.Float32) for c in feature_cols])
    del df
    gc.collect()

    X = df_x.to_numpy()
    del df_x
    gc.collect()
    return X, y


def evaluate_split(
    model: xgb.XGBClassifier,
    split_name: str,
    parquet_path: str,
    feature_cols: list[str],
    benign_reference_path: str | None = None,
) -> dict:
    """
    Đánh giá mô hình trên một tập dữ liệu bằng Polars (theo từng batch nếu tập lớn để tiết kiệm RAM),
    tính toán Precision, Recall, F1-score, và PR-AUC.
    """
    t0 = time.time()
    lf = pl.scan_parquet(parquet_path)
    total_rows = lf.select(pl.len()).collect(engine="streaming").item()

    y_true_list = []
    y_prob_list = []

    batch_size = 2_000_000
    for offset in range(0, total_rows, batch_size):
        df_batch = (
            lf.slice(offset, batch_size)
            .select(feature_cols + ["label"])
            .collect(engine="streaming")
        )
        y_batch = df_batch["label"].to_numpy().astype(np.int8, copy=False)
        X_batch = (
            df_batch.drop("label")
            .with_columns([pl.col(c).cast(pl.Float32) for c in feature_cols])
            .to_numpy()
        )
        del df_batch

        prob_batch = model.predict_proba(X_batch)[:, 1].astype(np.float32)
        del X_batch

        y_true_list.append(y_batch)
        y_prob_list.append(prob_batch)
        gc.collect()

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    y_pred = (y_prob >= 0.5).astype(np.int8)
    del y_true_list, y_prob_list
    gc.collect()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    pr_auc = float(average_precision_score(y_true, y_prob))

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    res = {
        "split": split_name,
        "rows": int(total_rows),
        "positive_attack": n_pos,
        "negative_benign": n_neg,
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

    # Nếu tập chỉ có lớp 1 (như Test_unseen chỉ chứa Bot & Infilteration),
    # tính thêm PR-AUC kết hợp với Benign của Test_seen để đánh giá khả năng phân biệt Unseen vs Benign
    if n_neg == 0 and benign_reference_path is not None and os.path.exists(benign_reference_path):
        df_benign = (
            pl.scan_parquet(benign_reference_path)
            .filter(pl.col("label") == 0)
            .select(feature_cols + ["label"])
            .head(n_pos)  # Lấy cân bằng 1:1 hoặc tương đương với số mẫu Unseen
            .collect(engine="streaming")
        )
        X_ben = (
            df_benign.drop("label")
            .with_columns([pl.col(c).cast(pl.Float32) for c in feature_cols])
            .to_numpy()
        )
        prob_ben = model.predict_proba(X_ben)[:, 1].astype(np.float32)
        pred_ben = (prob_ben >= 0.5).astype(np.int8)
        del df_benign, X_ben
        gc.collect()

        y_comb = np.concatenate([y_true, np.zeros(len(prob_ben), dtype=np.int8)])
        prob_comb = np.concatenate([y_prob, prob_ben])
        pred_comb = np.concatenate([y_pred, pred_ben])

        res["combined_with_benign_precision"] = float(precision_score(y_comb, pred_comb, zero_division=0))
        res["combined_with_benign_f1"] = float(f1_score(y_comb, pred_comb, zero_division=0))
        res["combined_with_benign_pr_auc"] = float(average_precision_score(y_comb, prob_comb))

        # Chi tiết Recall theo từng loại tấn công Unseen (Bot vs Infilteration)
        attacks = pl.scan_parquet(parquet_path).select("Attack").collect(engine="streaming")["Attack"].to_list()
        attack_arr = np.array(attacks)
        per_attack_recall = {}
        for atk in np.unique(attack_arr):
            mask = attack_arr == atk
            per_attack_recall[str(atk)] = {
                "count": int(mask.sum()),
                "detected_tp": int(y_pred[mask].sum()),
                "recall": float(y_pred[mask].mean()),
                "mean_prob": float(y_prob[mask].mean()),
            }
        res["unseen_attack_breakdown"] = per_attack_recall

    del y_true, y_prob, y_pred
    gc.collect()
    return res


def main():
    start_time = time.time()
    proc_dir = os.path.join("data", "processed")

    train_path = os.path.join(proc_dir, "edges_train.parquet")
    val_path = os.path.join(proc_dir, "edges_val.parquet")
    test_seen_path = os.path.join(proc_dir, "edges_test_seen.parquet")
    test_unseen_path = os.path.join(proc_dir, "edges_test_unseen.parquet")

    feature_cols = get_feature_cols(train_path)
    print("=" * 80)
    print("HUẤN LUYỆN BASELINE PHI ĐỒ THỊ (B0): XGBOOST CLASSIFIER")
    print("=" * 80)
    print(f"- Số lượng cột đặc trưng (X) : {len(feature_cols)} cột")
    print(f"- Các cột đã loại bỏ khỏi X  : {NON_FEATURE_COLS}")
    print(f"- Biến mục tiêu (y)          : label (0 = Benign, 1 = Attack)\n")

    # 1. Đọc dữ liệu Train
    print("[1/3] Đang nạp dữ liệu huấn luyện từ 'data/processed/edges_train.parquet' (10,000,000 dòng đại diện)...")
    t0 = time.time()
    X_train, y_train = load_xy_polars(train_path, feature_cols, max_rows=10_000_000)

    print(
        f"      -> Đã nạp X_train: {X_train.shape} ({X_train.dtype}), "
        f"y_train: {y_train.shape} trong {time.time() - t0:.1f}s\n"
    )

    # 2. Khởi tạo và huấn luyện mô hình XGBoost
    print("[2/3] Đang huấn luyện mô hình XGBClassifier (n_estimators=100, max_depth=6, tree_method='hist')...")
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"      -> Huấn luyện hoàn tất trong {train_time:.1f}s.")

    # Lưu model và giải phóng ngay X_train, y_train khỏi RAM
    os.makedirs(os.path.join("models", "baselines"), exist_ok=True)
    model_path = os.path.join("models", "baselines", "xgboost_baseline_b0.json")
    model.save_model(model_path)
    print(f"      -> Đã lưu mô hình tại '{model_path}'. Đang giải phóng X_train, y_train khỏi RAM...\n")
    del X_train, y_train
    gc.collect()

    # 3. Đánh giá trên 3 tập: Validation, Test_seen, Test_unseen
    print("[3/3] Đang dự đoán (predict & predict_proba) và tính chỉ số trên 3 tập Validation, Test_seen, Test_unseen...")
    eval_splits = [
        ("Validation", val_path),
        ("Test_seen", test_seen_path),
        ("Test_unseen (Unseen Attack)", test_unseen_path),
    ]

    all_metrics = []
    for name, path in eval_splits:
        print(f"      -> Đang đánh giá tập {name} ({path})...")
        m = evaluate_split(
            model,
            name,
            path,
            feature_cols,
            benign_reference_path=test_seen_path,
        )
        all_metrics.append(m)
        gc.collect()

    # Lưu kết quả ra JSON
    os.makedirs(os.path.join("results", "metrics"), exist_ok=True)
    metrics_json_path = os.path.join("results", "metrics", "baseline_b0_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # In bảng tổng hợp kết quả
    print("\n" + "=" * 86)
    print("KẾT QUẢ ĐÁNH GIÁ BASELINE PHI ĐỒ THỊ (B0 - XGBOOST)")
    print("=" * 86)
    print(
        f"{'Tập dữ liệu':<24} | {'Số dòng':>11} | {'Precision':>10} | {'Recall':>10} | {'F1-score':>10} | {'PR-AUC':>10}"
    )
    print("-" * 86)
    for m in all_metrics:
        print(
            f"{m['split']:<24} | {m['rows']:>11,} | "
            f"{m['precision']:>10.4f} | {m['recall']:>10.4f} | "
            f"{m['f1']:>10.4f} | {m['pr_auc']:>10.4f}"
        )
    print("-" * 86)

    # In phân tích sâu cho tập Test_unseen (Unseen: Bot & Infilteration)
    unseen_m = all_metrics[-1]
    print("\n[PHÂN TÍCH CHI TIẾT TẬP TEST_UNSEEN (UNSEEN: BOT & INFILTERATION)]")
    print(f"- Tổng số mẫu Unseen (label=1)            : {unseen_m['rows']:,} dòng (Benign=0)")
    print(f"- Phát hiện đúng là tấn công (TP)         : {unseen_m['TP']:,} / {unseen_m['rows']:,} dòng (Recall = {unseen_m['recall']*100:.2f}%)")
    print(f"- Bỏ lọt thành bình thường (FN - Missed)  : {unseen_m['FN']:,} / {unseen_m['rows']:,} dòng (Miss Rate = {(1 - unseen_m['recall'])*100:.2f}%)")
    print(f"- Xác suất dự đoán tấn công trung bình    : {unseen_m['mean_pred_prob']:.4f}")
    if "combined_with_benign_pr_auc" in unseen_m:
        print(
            f"- PR-AUC (khi ghép 1:1 với Benign thực tế): {unseen_m['combined_with_benign_pr_auc']:.4f} "
            f"(F1 = {unseen_m['combined_with_benign_f1']:.4f}, Precision = {unseen_m['combined_with_benign_precision']:.4f})"
        )
    if "unseen_attack_breakdown" in unseen_m:
        print("- Chi tiết theo từng họ tấn công Unseen:")
        for atk, info in unseen_m["unseen_attack_breakdown"].items():
            print(
                f"   + {atk:<10}: {info['detected_tp']:>6,}/{info['count']:<6,} phát hiện được "
                f"| Recall = {info['recall']*100:>6.2f}% | Mean Prob = {info['mean_prob']:.4f}"
            )
    print("=" * 86)
    print(f"Tổng thời gian chạy toàn bộ script: {time.time() - start_time:.1f} giây")


if __name__ == "__main__":
    main()
