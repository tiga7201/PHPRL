import torch
import torch.nn as nn


class PGNNPhase1(nn.Module):
    """
    Phase-1 PGNN:
    input  = [F_q, f_q_p, status]
    output = delta_F_hat
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PGNNPhase2(nn.Module):
    """
    Phase-2 PGNN:
    input  = [delta_F_hat_1, difficulty, automation, workload, fatigue, status]
    output = correction factor r in [-1, 1]
    """
    def __init__(self, hidden_dim: int = 32, max_correction: float = 0.5):
        super().__init__()
        self.max_correction = max_correction

        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        raw = self.net(x).squeeze(-1)
        r = torch.tanh(raw)
        return r