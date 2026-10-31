import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, GINEConv

NUM_TOTAL_NODES = 284_339
NODE_INIT_DIM = 16
HIDDEN_DIM = 64
NUM_HEADS = 2
EDGE_ATTR_DIM = 41
NON_FEATURE_COLS = ["src_node_id", "dst_node_id", "label", "Attack", "Dataset"]


def compute_inductive_node_features(
    edge_index: torch.Tensor, num_nodes: int = NUM_TOTAL_NODES
) -> torch.Tensor:
    """
    Khởi tạo đặc trưng đỉnh quy nạp 16 chiều (Inductive Structural Node Initialization)
    từ cấu trúc đồ thị (edge_index) mà không phụ thuộc định danh tĩnh (Node ID) hay nhãn.
    """
    num_edges = max(1, edge_index.size(1))
    src, dst = edge_index[0], edge_index[1]

    out_deg = torch.bincount(src, minlength=num_nodes).float()
    in_deg = torch.bincount(dst, minlength=num_nodes).float()
    tot_deg = out_deg + in_deg

    active_nodes = max(1, int((tot_deg > 0).sum().item()))
    avg_deg = float(num_edges) / float(active_nodes)

    rel_out = out_deg / (avg_deg + 1e-5)
    rel_in = in_deg / (avg_deg + 1e-5)
    rel_tot = tot_deg / (avg_deg + 1e-5)

    log_rel_out = torch.log1p(rel_out)
    log_rel_in = torch.log1p(rel_in)
    log_rel_tot = torch.log1p(rel_tot)

    out_ratio = out_deg / (tot_deg + 1.0)
    in_ratio = in_deg / (tot_deg + 1.0)

    dst_in_per_edge = log_rel_in[dst]
    src_out_per_edge = log_rel_out[src]

    nbr_in_sum = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    nbr_in_sum.scatter_add_(0, src, dst_in_per_edge)
    avg_dst_in = nbr_in_sum / (out_deg + 1.0)

    nbr_out_sum = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    nbr_out_sum.scatter_add_(0, dst, src_out_per_edge)
    avg_src_out = nbr_out_sum / (in_deg + 1.0)

    ones = torch.ones_like(out_deg)
    return torch.stack(
        [
            ones,
            log_rel_out,
            log_rel_in,
            log_rel_tot,
            out_ratio,
            in_ratio,
            avg_dst_in,
            avg_src_out,
            torch.tanh(rel_out * 0.25),
            torch.tanh(rel_in * 0.25),
            torch.tanh(rel_tot * 0.25),
            torch.clamp(log_rel_tot - 3.5, min=0.0),
            torch.clamp(avg_dst_in - 3.5, min=0.0),
            torch.clamp(avg_src_out - 3.5, min=0.0),
            (log_rel_out + avg_dst_in) * 0.5,
            (log_rel_in + avg_src_out) * 0.5,
        ],
        dim=-1,
    )


class GNNBaselineG0(nn.Module):
    """Mô hình G0 (Transductive GCN + nn.Embedding)."""

    def __init__(
        self,
        num_nodes: int = NUM_TOTAL_NODES,
        embedding_dim: int = HIDDEN_DIM,
        hidden_dim: int = HIDDEN_DIM,
        edge_attr_dim: int = EDGE_ATTR_DIM,
    ):
        super().__init__()
        self.node_emb = nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
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
        x = self.node_emb(n_id)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        src_idx, dst_idx = edge_label_index[0], edge_label_index[1]
        edge_repr = torch.cat([x[src_idx], x[dst_idx], target_edge_attr], dim=-1)
        return self.classifier(edge_repr).squeeze(-1)


class GNNInductiveG1(nn.Module):
    """Mô hình G1 (Inductive GINEConv - Sum Aggregation + BatchNorm1d)."""

    def __init__(
        self,
        node_init_dim: int = NODE_INIT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        edge_attr_dim: int = EDGE_ATTR_DIM,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_init_dim, hidden_dim)
        gin_mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GINEConv(nn=gin_mlp1, edge_dim=edge_attr_dim, train_eps=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        gin_mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = GINEConv(nn=gin_mlp2, edge_dim=edge_attr_dim, train_eps=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        classifier_in_dim = hidden_dim * 2 + edge_attr_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_label_index: torch.Tensor,
        target_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        e_sub = torch.sign(edge_attr) * torch.log1p(torch.abs(edge_attr))
        e_tar = torch.sign(target_edge_attr) * torch.log1p(torch.abs(target_edge_attr))
        h = F.relu(self.node_proj(x))

        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        bi_edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
        bi_edge_attr = torch.cat([e_sub, e_sub], dim=0)

        h1 = F.relu(self.bn1(self.conv1(h, bi_edge_index, edge_attr=bi_edge_attr)))
        h1 = F.dropout(h1, p=0.2, training=self.training)
        h2 = F.relu(self.bn2(self.conv2(h1, bi_edge_index, edge_attr=bi_edge_attr)))

        src_idx, dst_idx = edge_label_index[0], edge_label_index[1]
        edge_repr = torch.cat([h2[src_idx], h2[dst_idx], e_tar], dim=-1)
        return self.classifier(edge_repr).squeeze(-1)


class GNNInductiveGATv2(nn.Module):
    """
    Mô hình đề xuất G1_v2 (Inductive Edge-Aware GATv2 + LayerNorm + Dual-Pathway)
    hỗ trợ đồng thời:
      - Phân loại nhị phân quy nạp (T=1)
      - Định lượng bất định nhận thức 3 nhánh (Three-way MC Dropout UQ, T=20)
      - Nhận dạng Tập Mở (Open-Set Recognition - OSR) qua năng lượng bất đồng hai nhánh (Dual-Pathway Discrepancy).
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
            dropout=0.15,
            add_self_loops=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=out_per_head,
            heads=heads,
            concat=True,
            edge_dim=edge_attr_dim,
            dropout=0.15,
            add_self_loops=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.mc_node_drop = nn.Dropout(p=0.20)
        self.topo_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.20),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.flow_head = nn.Sequential(
            nn.Linear(edge_attr_dim + hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.20),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _scale_edge_attr(self, e: torch.Tensor) -> torch.Tensor:
        e_comp = torch.sign(e) * torch.log1p(torch.abs(e))
        return self.edge_norm(e_comp)

    def encode_subgraph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_label_index: torch.Tensor,
        target_edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.gelu(self.node_proj(x))
        e_sub = self._scale_edge_attr(edge_attr)
        e_target = self._scale_edge_attr(target_edge_attr)

        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        bi_edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
        bi_edge_attr = torch.cat([e_sub, e_sub], dim=0)
        return h, bi_edge_index, bi_edge_attr, edge_label_index, e_target

    def forward_dual_logits(
        self,
        h: torch.Tensor,
        bi_edge_index: torch.Tensor,
        bi_edge_attr: torch.Tensor,
        edge_label_index: torch.Tensor,
        e_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.gat1(h, bi_edge_index, edge_attr=bi_edge_attr)
        h1 = F.gelu(self.norm1(h1))
        h1 = self.mc_node_drop(h1)

        h2 = self.gat2(h1, bi_edge_index, edge_attr=bi_edge_attr)
        h2 = F.gelu(self.norm2(h2 + h1))
        h2 = self.mc_node_drop(h2)

        src_idx, dst_idx = edge_label_index[0], edge_label_index[1]
        node_pair = torch.cat([h2[src_idx], h2[dst_idx]], dim=-1)
        edge_repr = torch.cat([node_pair, e_target], dim=-1)

        logit_topo = self.topo_head(node_pair).squeeze(-1)
        logit_flow = self.flow_head(edge_repr).squeeze(-1)
        return logit_topo, logit_flow

    def forward_single_mc(
        self,
        h: torch.Tensor,
        bi_edge_index: torch.Tensor,
        bi_edge_attr: torch.Tensor,
        edge_label_index: torch.Tensor,
        e_target: torch.Tensor,
    ) -> torch.Tensor:
        logit_topo, logit_flow = self.forward_dual_logits(
            h, bi_edge_index, bi_edge_attr, edge_label_index, e_target
        )
        if self.training:
            gate = torch.empty_like(logit_topo).bernoulli_(0.55)
            logits = gate * logit_topo + (1.0 - gate) * logit_flow
        else:
            logits = 0.55 * logit_topo + 0.45 * logit_flow
        return torch.sigmoid(logits)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_label_index: torch.Tensor,
        target_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h, bi_ei, bi_ea, eli, e_target = self.encode_subgraph(
            x, edge_index, edge_attr, edge_label_index, target_edge_attr
        )
        logit_topo, logit_flow = self.forward_dual_logits(h, bi_ei, bi_ea, eli, e_target)
        if self.training:
            gate = torch.empty_like(logit_topo).bernoulli_(0.55)
            return gate * logit_topo + (1.0 - gate) * logit_flow
        return 0.55 * logit_topo + 0.45 * logit_flow
