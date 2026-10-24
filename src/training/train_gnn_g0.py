import gc
import os
import sys
import time
import warnings
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore", category=FutureWarning)

# Đảm bảo console trên Windows hiển thị đúng tiếng Việt (UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


NUM_TOTAL_NODES = 284_339
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
EDGE_ATTR_DIM = 41
BATCH_SIZE = 4096
NUM_NEIGHBORS = [10, 10]
NON_FEATURE_COLS = ["src_node_id", "dst_node_id", "label", "Attack", "Dataset"]


class GNNBaselineG0(nn.Module):
    """
    GNN Baseline (G0 - GCN) cho bài toán phát hiện xâm nhập mạng (Edge-level Binary Classification):
    - Node features: nn.Embedding(num_embeddings=284339, embedding_dim=64)
    - Message Passing: 2 lớp GCNConv cập nhật biểu diễn nút (node embeddings)
    - Edge Classifier (MLP): Nhận đầu vào là phép ghép (concatenate) [node_src || node_dst || edge_attr]
      có kích thước (64 + 64 + 41 = 169) -> dự đoán logit cho phân loại nhị phân (0: Benign, 1: Attack).
    """

    def __init__(
        self,
        num_nodes: int = NUM_TOTAL_NODES,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        edge_attr_dim: int = EDGE_ATTR_DIM,
    ):
        super().__init__()
        # Khởi tạo đặc trưng nút bằng lớp Embedding vì IP gốc không có đặc trưng nội tại
        self.node_emb = nn.Embedding(
            num_embeddings=num_nodes,
            embedding_dim=embedding_dim,
        )

        # 2 lớp GCNConv thực hiện Message Passing
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Classifier (MLP) kết hợp: [node_src (64) || node_dst (64) || edge_attr (41)] -> 1 logit
        classifier_in_dim = hidden_dim * 2 + edge_attr_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        n_id: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        target_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Tra cứu embedding ban đầu cho các node xuất hiện trong subgraph của mini-batch
        x = self.node_emb(n_id)  # [num_sampled_nodes, 64]

        # 2. Message Passing qua 2 lớp GCNConv trên đồ thị con (edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)  # [num_sampled_nodes, 64]

        # 3. Trích xuất biểu diễn của node nguồn và node đích cho các cạnh mục tiêu trong batch
        # Lưu ý: Trong LinkNeighborLoader, edge_label_index chứa chỉ số cục bộ (local index)
        # tương ứng với tensor x của subgraph hiện tại.
        src_idx, dst_idx = edge_label_index[0], edge_label_index[1]
        node_src = x[src_idx]  # [batch_size, 64]
        node_dst = x[dst_idx]  # [batch_size, 64]

        # 4. Kết hợp (Concatenate): [node_src, node_dst, edge_attr] -> [batch_size, 169]
        edge_repr = torch.cat([node_src, node_dst, target_edge_attr], dim=-1)

        # 5. Dự đoán logit (dùng với BCEWithLogitsLoss)
        logits = self.classifier(edge_repr).squeeze(-1)  # [batch_size]
        return logits


def load_graph_data(parquet_path: str, max_edges: int | None = None) -> Data:
    """
    Dùng Polars đọc edges_train.parquet và đóng gói thành đối tượng torch_geometric.data.Data.
    Tối ưu bộ nhớ bằng cách giải phóng từng cột ngay sau khi chuyển sang Tensor.
    """
    t0 = time.time()
    lf = pl.scan_parquet(parquet_path)
    total_edges = lf.select(pl.len()).collect(engine="streaming").item()
    schema_cols = lf.collect_schema().names()
    feature_cols = [c for c in schema_cols if c not in NON_FEATURE_COLS]

    if max_edges is not None and total_edges > max_edges:
        print(
            f"      [*] File có {total_edges:,} cạnh. Đang nạp {max_edges:,} cạnh để đảm bảo an toàn RAM cho CSR Graph..."
        )
        lf = lf.head(max_edges)
        loaded_edges = max_edges
    else:
        loaded_edges = total_edges

    print(f"      -> Đang đọc src_node_id, dst_node_id và label ({loaded_edges:,} cạnh)...")
    df_topo = lf.select(["src_node_id", "dst_node_id", "label"]).collect(engine="streaming")

    # PyG yêu cầu edge_index có kiểu torch.long (int64) và kích thước [2, num_edges]
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

    print(f"      -> Đang đọc {len(feature_cols)} cột đặc trưng NetFlow làm edge_attr...")
    df_feat = (
        lf.select([pl.col(c).cast(pl.Float32) for c in feature_cols])
        .collect(engine="streaming")
    )
    edge_attr = torch.from_numpy(df_feat.to_numpy())
    del df_feat
    gc.collect()

    # Khởi tạo đối tượng PyG Data
    data = Data(
        num_nodes=NUM_TOTAL_NODES,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )
    print(
        f"      -> Đã khởi tạo PyG Data thành công trong {time.time() - t0:.1f}s: "
        f"num_nodes={data.num_nodes:,}, edge_index={tuple(data.edge_index.shape)}, "
        f"edge_attr={tuple(data.edge_attr.shape)}, y={tuple(data.y.shape)}\n"
    )
    return data


def main():
    start_time = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_parquet = os.path.join("data", "edges_train.parquet")

    print("=" * 80)
    print("GNN BASELINE (G0 - GCN) - KIỂM THỬ MINI-BATCH & FORWARD PASS (5 BATCHES)")
    print("=" * 80)
    print(f"- Thiết bị thực thi (Device)   : {device}")
    print(f"- Tổng số Node toàn mạng       : {NUM_TOTAL_NODES:,} (Embedding: {NUM_TOTAL_NODES} x {EMBEDDING_DIM})")
    print(f"- Số chiều đặc trưng cạnh      : {EDGE_ATTR_DIM} cột NetFlow")
    print(f"- Cấu hình LinkNeighborLoader  : num_neighbors={NUM_NEIGHBORS}, batch_size={BATCH_SIZE}, shuffle=True\n")

    # 1. Khởi tạo Đồ thị từ edges_train.parquet
    print("[1/3] Đang nạp đồ thị từ 'data/edges_train.parquet'...")
    try:
        # Nạp 5,000,000 cạnh đầu tiên (đã được shuffle sẵn ở bước split) nếu toàn bộ 53M cạnh vượt quá RAM khi tạo CSR
        # Thử nạp 10,000,000 cạnh trước để vừa đại diện đồ thị lớn vừa tránh tràn RAM khi tạo ma trận kề CSC/CSR của NeighborSampler
        data = load_graph_data(train_parquet, max_edges=10_000_000)
    except MemoryError:
        gc.collect()
        print("      [!] Chuyển sang nạp 2,000,000 cạnh để tiết kiệm RAM...")
        data = load_graph_data(train_parquet, max_edges=2_000_000)

    # 2. Cấu hình LinkNeighborLoader
    print("[2/3] Đang cấu hình LinkNeighborLoader (num_neighbors=[10, 10], batch_size=4096, shuffle=True)...")
    t0 = time.time()
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        edge_label_index=data.edge_index,
        edge_label=data.y,
        shuffle=True,
    )
    print(f"      -> Khởi tạo LinkNeighborLoader hoàn tất trong {time.time() - t0:.2f}s.\n")

    # 3. Khởi tạo mô hình G0 (GCN), hàm mất mát BCEWithLogitsLoss và bộ tối ưu Adam
    model = GNNBaselineG0(
        num_nodes=NUM_TOTAL_NODES,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        edge_attr_dim=EDGE_ATTR_DIM,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("[3/3] Đang chạy thử Forward & Backward Pass trên 5 batches đầu tiên...")
    print("-" * 80)
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:
            break

        b_t0 = time.time()
        batch = batch.to(device)
        optimizer.zero_grad()

        # Lấy đặc trưng 41 cột NetFlow (edge_attr) của đúng các cạnh mục tiêu (seed edges) trong batch
        # LinkNeighborLoader lưu chỉ số gốc của các cạnh mục tiêu trong `batch.input_id`
        target_edge_attr = data.edge_attr[batch.input_id.cpu()].to(device)
        target_labels = batch.edge_label.float()

        # Forward pass qua GCN (2 lớp) + MLP Classifier (node_src || node_dst || edge_attr)
        logits = model(
            n_id=batch.n_id,
            edge_index=batch.edge_index,
            edge_label_index=batch.edge_label_index,
            target_edge_attr=target_edge_attr,
        )

        loss = criterion(logits, target_labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == target_labels).float().mean().item()

        print(
            f"Batch {batch_idx + 1}/5 | "
            f"Loss: {loss.item():.6f} | "
            f"Acc: {acc * 100:5.2f}% | "
            f"Subgraph Nodes: {batch.n_id.size(0):>6,} | "
            f"Subgraph Edges: {batch.edge_index.size(1):>7,} | "
            f"Target Edges: {logits.size(0):>4} | "
            f"Time: {time.time() - b_t0:.2f}s"
        )

    print("-" * 80)
    print(
        f"[+] Xác nhận hoàn tất 5 batches thành công (Không lỗi chiều tensor, không bị OOM)! "
        f"Tổng thời gian: {time.time() - start_time:.1f}s"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
