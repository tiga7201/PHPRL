import torch
import torch.nn as nn


class DiscreteQCritic(nn.Module):
    """
    For each hyperedge/action, output a Q-value.
    Uses the same action-wise feature construction as the actor.
    """

    def __init__(
        self,
        op_dim=7,
        machine_dim=5,
        worker_dim=7,
        edge_dim=3,
        hidden_dim=128,
    ):
        super().__init__()

        local_dim = op_dim + machine_dim + worker_dim + edge_dim
        global_dim = op_dim + machine_dim + worker_dim + edge_dim
        input_dim = local_dim + global_dim

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
        edge_links = torch.as_tensor(graph_state["edge_links"], dtype=torch.long)

        op_idx = edge_links[:, 0]
        machine_idx = edge_links[:, 1]
        worker_idx = edge_links[:, 2]

        edge_op_feat = op_features[op_idx]
        edge_machine_feat = machine_features[machine_idx]
        edge_worker_feat = worker_features[worker_idx]

        local_action_inputs = torch.cat(
            [edge_op_feat, edge_machine_feat, edge_worker_feat, edge_features],
            dim=-1
        )

        global_op = op_features.mean(dim=0, keepdim=True)
        global_machine = machine_features.mean(dim=0, keepdim=True)
        global_worker = worker_features.mean(dim=0, keepdim=True)
        global_edge = edge_features.mean(dim=0, keepdim=True)

        global_feat = torch.cat(
            [global_op, global_machine, global_worker, global_edge],
            dim=-1
        )
        global_feat = global_feat.repeat(local_action_inputs.shape[0], 1)

        action_inputs = torch.cat([local_action_inputs, global_feat], dim=-1)

        q_values = self.net(action_inputs).squeeze(-1)  # [num_edges]
        return q_values
