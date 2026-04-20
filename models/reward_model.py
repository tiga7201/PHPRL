import torch
import torch.nn as nn

from models.shypergnn_full import SHyperGNNFull
from rl.embedding_cache import extract_state_action_embedding


class RewardModel(nn.Module):
    """
    Preference-based reward model.

    Two modes:
      1) score_state_action(graph_state, chosen_edge_idx)
      2) score_sa_embedding(sa_embedding)
    """

    def __init__(self, hidden_dim=64, num_layers=2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoder = SHyperGNNFull(hidden_dim=hidden_dim, num_layers=num_layers)

        input_dim = hidden_dim * 7  # op + machine + worker + edge + graph(3d)

        self.step_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def score_sa_embedding(self, sa_embedding):
        device = next(self.parameters()).device
        if not torch.is_tensor(sa_embedding):
            sa_embedding = torch.as_tensor(sa_embedding, dtype=torch.float32, device=device)
        else:
            sa_embedding = sa_embedding.to(device)

        return self.step_head(sa_embedding).squeeze(-1)

    def score_state_action(self, graph_state, chosen_edge_idx: int):
        sa_emb = extract_state_action_embedding(self.encoder, graph_state, chosen_edge_idx)
        return self.score_sa_embedding(sa_emb)

    def score_trajectory(self, trajectory):
        total = None

        # fast path: cached embeddings
        if trajectory.sa_embeddings is not None:
            for sa_emb in trajectory.sa_embeddings:
                s = self.score_sa_embedding(sa_emb)
                total = s if total is None else total + s
        else:
            for graph_state, edge_idx in zip(trajectory.graph_states, trajectory.edge_indices):
                s = self.score_state_action(graph_state, edge_idx)
                total = s if total is None else total + s

        if total is None:
            device = next(self.parameters()).device
            total = torch.tensor(0.0, device=device)

        return total

    def score_trajectory_batch(self, trajectories):
        """
        Score a batch of trajectories.
        Uses cached sa_embeddings if available.
        Returns:
            scores: tensor [B]
        """
        scores = []
        for traj in trajectories:
            scores.append(self.score_trajectory(traj))
        return torch.stack(scores, dim=0)