import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models.shypergnn import SHyperGNN


class SHyperActor(nn.Module):
    """
    Actor based on SHyperGNN embeddings.

    For each action(edge), score:
    [op_emb, machine_emb, worker_emb, edge_emb, graph_emb]
    """

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.encoder = SHyperGNN(hidden_dim=hidden_dim)

        input_dim = hidden_dim * 7  # op + machine + worker + edge + graph(3*hidden)

        self.policy_head = nn.Sequential(
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
        valid_action_mask = torch.as_tensor(
            graph_state["valid_action_mask"], dtype=torch.float32, device=device
        )

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

        logits = self.policy_head(action_input).squeeze(-1)

        invalid_mask = valid_action_mask < 0.5
        logits = logits.masked_fill(invalid_mask, -1e9)

        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "valid_action_mask": valid_action_mask,
        }

    def sample_action(self, graph_state):
        output = self.forward(graph_state)
        probs = output["probs"]

        dist = Categorical(probs=probs)
        edge_idx = dist.sample()
        log_prob = dist.log_prob(edge_idx)

        edge_idx_int = int(edge_idx.item())
        action = self.edge_idx_to_action(graph_state, edge_idx_int)

        return {
            "edge_idx": edge_idx_int,
            "action": action,
            "log_prob": log_prob,
            "probs": probs,
        }

    def select_greedy_action(self, graph_state):
        output = self.forward(graph_state)
        probs = output["probs"]
        edge_idx = int(torch.argmax(probs).item())
        action = self.edge_idx_to_action(graph_state, edge_idx)

        return {
            "edge_idx": edge_idx,
            "action": action,
            "probs": probs,
        }

    @staticmethod
    def edge_idx_to_action(graph_state, edge_idx: int):
        return graph_state["edge_to_action"][edge_idx]