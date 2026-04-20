import torch
import torch.nn as nn


class SHyperGNN(nn.Module):
    """
    Simplified scheduling hypergraph neural network (v1).

    Stage 1: node -> edge
    Stage 2: edge -> node (mean aggregation)

    Inputs:
        graph_state dict with:
        - op_features: [num_ops, op_dim]
        - machine_features: [num_machines, machine_dim]
        - worker_features: [num_workers, worker_dim]
        - edge_features: [num_edges, edge_dim]
        - edge_links: [num_edges, 3]

    Outputs:
        dict with:
        - op_emb: [num_ops, hidden_dim]
        - machine_emb: [num_machines, hidden_dim]
        - worker_emb: [num_workers, hidden_dim]
        - edge_emb: [num_edges, hidden_dim]
        - graph_emb: [3 * hidden_dim]
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

        # stage 1: node -> edge
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # stage 2: edge -> node
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

        # initial projected node/edge representations
        op_h0 = self.op_proj(op_features)                  # [num_ops, d]
        machine_h0 = self.machine_proj(machine_features)   # [num_machines, d]
        worker_h0 = self.worker_proj(worker_features)      # [num_workers, d]
        edge_h0 = self.edge_proj(edge_features)            # [num_edges, d]

        # ---------- Stage 1: node -> edge ----------
        edge_input = torch.cat(
            [
                op_h0[op_idx],
                machine_h0[machine_idx],
                worker_h0[worker_idx],
                edge_h0,
            ],
            dim=-1,
        )  # [num_edges, 4d]

        edge_emb = self.edge_mlp(edge_input)  # [num_edges, d]

        # ---------- Stage 2: edge -> node ----------
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

        # ---------- Graph embedding ----------
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
        """
        Mean aggregation of incident edge embeddings to each node.
        """
        device = edge_emb.device
        hidden_dim = edge_emb.shape[1]

        agg = torch.zeros((num_nodes, hidden_dim), dtype=edge_emb.dtype, device=device)
        count = torch.zeros((num_nodes, 1), dtype=edge_emb.dtype, device=device)

        agg.index_add_(0, node_indices, edge_emb)

        ones = torch.ones((edge_emb.shape[0], 1), dtype=edge_emb.dtype, device=device)
        count.index_add_(0, node_indices, ones)

        agg = agg / count.clamp(min=1.0)
        return agg
