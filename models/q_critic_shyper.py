import torch
import torch.nn as nn

from models.shypergnn import SHyperGNN


class SHyperQCritic(nn.Module):
    """
    Q critic based on SHyperGNN embeddings.

    For each action(edge), output:
    Q(s, a) from [op_emb, machine_emb, worker_emb, edge_emb, graph_emb]
    """

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.encoder = SHyperGNN(hidden_dim=hidden_dim)

        input_dim = hidden_dim * 7  # op + machine + worker + edge + graph(3*hidden)

        self.q_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_state):
        device = next(self.parameters()).device

        enc = self.encoder(graph_state)

        edge_links = torch.as_tensor(graph_state["edge_links"], dtype=torch.long, device=device)

        op_idx = edge_links[:, 0]
        machine_idx = edge_links[:, 1]
        worker_idx = edge_links[:, 2]

        op_emb = enc["op_emb"][op_idx]
        machine_emb = enc["machine_emb"][machine_idx]
        worker_emb = enc["worker_emb"][worker_idx]
        edge_emb = enc["edge_emb"]
        graph_emb = enc["graph_emb"].unsqueeze(0).repeat(edge_emb.shape[0], 1)

        action_input = torch.cat(
            [op_emb, machine_emb, worker_emb, edge_emb, graph_emb],
            dim=-1,
        )

        q_values = self.q_head(action_input).squeeze(-1)  # [num_edges]
        return q_values