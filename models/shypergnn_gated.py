import torch
import torch.nn as nn


class SHyperGNNGated(nn.Module):
    """
    SHyperGNN v2:
    - Stage 1: node -> edge with gating
    - Stage 2: edge -> node with mean aggregation

    This keeps the second stage simple so we can isolate the effect of gating.
    """

    def __init__(
        self,
        op_dim=7,
        machine_dim=5,
        worker_dim=7,
        edge_dim=3,
        hidden_dim=64,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # input projections
        self.op_proj = nn.Linear(op_dim, hidden_dim)
        self.machine_proj = nn.Linear(machine_dim, hidden_dim)
        self.worker_proj = nn.Linear(worker_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # gates for node -> edge
        self.op_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.machine_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.worker_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # edge update after gated aggregation
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # node updates
        self.op_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.machine_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.worker_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, graph_state):
        device = next(self.parameters()).device

        op_features = torch.as_tensor(graph_state["op_features"], dtype=torch.float32, device=device)
        machine_features = torch.as_tensor(graph_state["machine_features"], dtype=torch.float32, device=device)
        worker_features = torch.as_tensor(graph_state["worker_features"], dtype=torch.float32, device=device)
        edge_features = torch.as_tensor(graph_state["edge_features"], dtype=torch.float32, device=device)
        edge_links = torch.as_tensor(graph_state["edge_links"], dtype=torch.long, device=device)

        op_idx = edge_links[:, 0]
        machine_idx = edge_links[:, 1]
        worker_idx = edge_links[:, 2]

        # initial projected representations
        op_h0 = self.op_proj(op_features)                  # [num_ops, d]
        machine_h0 = self.machine_proj(machine_features)   # [num_machines, d]
        worker_h0 = self.worker_proj(worker_features)      # [num_workers, d]
        edge_h0 = self.edge_proj(edge_features)            # [num_edges, d]

        # ---------- Stage 1: node -> edge with gating ----------
        op_edge_cat = torch.cat([op_h0[op_idx], edge_h0], dim=-1)
        machine_edge_cat = torch.cat([machine_h0[machine_idx], edge_h0], dim=-1)
        worker_edge_cat = torch.cat([worker_h0[worker_idx], edge_h0], dim=-1)

        g_op = torch.sigmoid(self.op_gate(op_edge_cat))
        g_machine = torch.sigmoid(self.machine_gate(machine_edge_cat))
        g_worker = torch.sigmoid(self.worker_gate(worker_edge_cat))

        gated_node_sum = (
            g_op * op_h0[op_idx]
            + g_machine * machine_h0[machine_idx]
            + g_worker * worker_h0[worker_idx]
        )

        edge_input = torch.cat([edge_h0, gated_node_sum], dim=-1)
        edge_emb = self.edge_mlp(edge_input)

        # ---------- Stage 2: edge -> node (mean aggregation) ----------
        op_agg = self._aggregate_to_nodes(
            num_nodes=op_h0.shape[0],
            node_indices=op_idx,
            edge_emb=edge_emb,
        )
        machine_agg = self._aggregate_to_nodes(
            num_nodes=machine_h0.shape[0],
            node_indices=machine_idx,
            edge_emb=edge_emb,
        )
        worker_agg = self._aggregate_to_nodes(
            num_nodes=worker_h0.shape[0],
            node_indices=worker_idx,
            edge_emb=edge_emb,
        )

        op_emb = self.op_update(torch.cat([op_h0, op_agg], dim=-1))
        machine_emb = self.machine_update(torch.cat([machine_h0, machine_agg], dim=-1))
        worker_emb = self.worker_update(torch.cat([worker_h0, worker_agg], dim=-1))

        # graph embedding
        graph_emb = torch.cat(
            [
                op_emb.mean(dim=0),
                machine_emb.mean(dim=0),
                worker_emb.mean(dim=0),
            ],
            dim=-1,
        )  # [3d]

        return {
            "op_emb": op_emb,
            "machine_emb": machine_emb,
            "worker_emb": worker_emb,
            "edge_emb": edge_emb,
            "graph_emb": graph_emb,
        }

    @staticmethod
    def _aggregate_to_nodes(num_nodes, node_indices, edge_emb):
        device = edge_emb.device
        hidden_dim = edge_emb.shape[1]

        agg = torch.zeros((num_nodes, hidden_dim), dtype=edge_emb.dtype, device=device)
        count = torch.zeros((num_nodes, 1), dtype=edge_emb.dtype, device=device)

        agg.index_add_(0, node_indices, edge_emb)

        ones = torch.ones((edge_emb.shape[0], 1), dtype=edge_emb.dtype, device=device)
        count.index_add_(0, node_indices, ones)

        agg = agg / count.clamp(min=1.0)
        return agg