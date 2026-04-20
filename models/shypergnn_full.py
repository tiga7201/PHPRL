import torch
import torch.nn as nn
import torch.nn.functional as F


class SHyperLayer(nn.Module):
    """
    One message passing layer for scheduling hypergraph:
    1) node -> edge with gating
    2) edge -> node with attention
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        d = hidden_dim

        # node -> edge gating
        self.op_gate = nn.Linear(2 * d, d)
        self.machine_gate = nn.Linear(2 * d, d)
        self.worker_gate = nn.Linear(2 * d, d)

        # edge update
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
        )

        # attention score for edge -> node
        self.op_attn = nn.Linear(2 * d, 1)
        self.machine_attn = nn.Linear(2 * d, 1)
        self.worker_attn = nn.Linear(2 * d, 1)

        # node update
        self.op_update = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.machine_update = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.worker_update = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

    def forward(self, op_h, machine_h, worker_h, edge_h, edge_links):
        op_idx = edge_links[:, 0]
        machine_idx = edge_links[:, 1]
        worker_idx = edge_links[:, 2]

        # ---------- Stage 1: node -> edge with gating ----------
        op_edge_cat = torch.cat([op_h[op_idx], edge_h], dim=-1)
        machine_edge_cat = torch.cat([machine_h[machine_idx], edge_h], dim=-1)
        worker_edge_cat = torch.cat([worker_h[worker_idx], edge_h], dim=-1)

        g_op = torch.sigmoid(self.op_gate(op_edge_cat))
        g_machine = torch.sigmoid(self.machine_gate(machine_edge_cat))
        g_worker = torch.sigmoid(self.worker_gate(worker_edge_cat))

        gated_sum = (
            g_op * op_h[op_idx]
            + g_machine * machine_h[machine_idx]
            + g_worker * worker_h[worker_idx]
        )

        edge_input = torch.cat([edge_h, gated_sum], dim=-1)
        new_edge_h = self.edge_mlp(edge_input)

        # residual edge update
        edge_h = edge_h + new_edge_h

        # ---------- Stage 2: edge -> node with attention ----------
        op_agg = self._attn_aggregate(
            node_h=op_h,
            node_indices=op_idx,
            edge_h=edge_h,
            attn_layer=self.op_attn,
        )
        machine_agg = self._attn_aggregate(
            node_h=machine_h,
            node_indices=machine_idx,
            edge_h=edge_h,
            attn_layer=self.machine_attn,
        )
        worker_agg = self._attn_aggregate(
            node_h=worker_h,
            node_indices=worker_idx,
            edge_h=edge_h,
            attn_layer=self.worker_attn,
        )

        # residual node updates
        op_h = op_h + self.op_update(torch.cat([op_h, op_agg], dim=-1))
        machine_h = machine_h + self.machine_update(torch.cat([machine_h, machine_agg], dim=-1))
        worker_h = worker_h + self.worker_update(torch.cat([worker_h, worker_agg], dim=-1))

        return op_h, machine_h, worker_h, edge_h

    @staticmethod
    def _attn_aggregate(node_h, node_indices, edge_h, attn_layer):
        """
        Attention aggregation from incident edges to nodes.
        Implemented per node for clarity and correctness.
        """
        device = edge_h.device
        num_nodes = node_h.shape[0]
        d = edge_h.shape[1]

        out = torch.zeros((num_nodes, d), dtype=edge_h.dtype, device=device)

        for node_id in range(num_nodes):
            mask = (node_indices == node_id)
            if not torch.any(mask):
                continue

            incident_edges = edge_h[mask]                  # [k, d]
            repeated_node = node_h[node_id].unsqueeze(0).repeat(incident_edges.shape[0], 1)
            attn_input = torch.cat([incident_edges, repeated_node], dim=-1)  # [k, 2d]

            scores = attn_layer(attn_input).squeeze(-1)   # [k]
            weights = F.softmax(scores, dim=0)            # [k]

            out[node_id] = torch.sum(weights.unsqueeze(-1) * incident_edges, dim=0)

        return out


class SHyperGNNFull(nn.Module):
    """
    Full SHyperGNN:
    - input projections
    - multiple SHyper layers
    - graph pooling
    """

    def __init__(
        self,
        op_dim=7,
        machine_dim=5,
        worker_dim=7,
        edge_dim=3,
        hidden_dim=64,
        num_layers=2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # input projections
        self.op_proj = nn.Linear(op_dim, hidden_dim)
        self.machine_proj = nn.Linear(machine_dim, hidden_dim)
        self.worker_proj = nn.Linear(worker_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.layers = nn.ModuleList([
            SHyperLayer(hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, graph_state):
        device = next(self.parameters()).device

        op_features = torch.as_tensor(graph_state["op_features"], dtype=torch.float32, device=device)
        machine_features = torch.as_tensor(graph_state["machine_features"], dtype=torch.float32, device=device)
        worker_features = torch.as_tensor(graph_state["worker_features"], dtype=torch.float32, device=device)
        edge_features = torch.as_tensor(graph_state["edge_features"], dtype=torch.float32, device=device)
        edge_links = torch.as_tensor(graph_state["edge_links"], dtype=torch.long, device=device)

        op_h = self.op_proj(op_features)
        machine_h = self.machine_proj(machine_features)
        worker_h = self.worker_proj(worker_features)
        edge_h = self.edge_proj(edge_features)

        for layer in self.layers:
            op_h, machine_h, worker_h, edge_h = layer(
                op_h, machine_h, worker_h, edge_h, edge_links
            )

        graph_emb = torch.cat(
            [
                op_h.mean(dim=0),
                machine_h.mean(dim=0),
                worker_h.mean(dim=0),
            ],
            dim=-1,
        )  # [3d]

        return {
            "op_emb": op_h,
            "machine_emb": machine_h,
            "worker_emb": worker_h,
            "edge_emb": edge_h,
            "graph_emb": graph_emb,
        }