import torch
import torch.nn as nn


class BaselineCritic(nn.Module):
    """
    A simple state-value critic.
    Uses pooled graph statistics as input.
    """

    def __init__(self, op_dim=7, machine_dim=5, worker_dim=7, edge_dim=3, hidden_dim=128):
        super().__init__()

        input_dim = op_dim + machine_dim + worker_dim + edge_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_state):
        op_features = torch.as_tensor(graph_state["op_features"], dtype=torch.float32)
        machine_features = torch.as_tensor(graph_state["machine_features"], dtype=torch.float32)
        worker_features = torch.as_tensor(graph_state["worker_features"], dtype=torch.float32)
        edge_features = torch.as_tensor(graph_state["edge_features"], dtype=torch.float32)

        op_pool = op_features.mean(dim=0)
        machine_pool = machine_features.mean(dim=0)
        worker_pool = worker_features.mean(dim=0)
        edge_pool = edge_features.mean(dim=0)

        state_feat = torch.cat([op_pool, machine_pool, worker_pool, edge_pool], dim=-1)
        value = self.net(state_feat).squeeze(-1)
        return value
