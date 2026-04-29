import torch

from models.pgnn import PGNNPhase1, PGNNPhase2


class PGNNPhase1Inference:
    def __init__(self, checkpoint_path: str, device: str = "cpu", hidden_dim: int = 32):
        self.device = device
        self.model = PGNNPhase1(hidden_dim=hidden_dim).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_delta_f(self, fatigue: float, f_p: int, status: int) -> float:
        x = torch.tensor(
            [[fatigue, (f_p - 1) / 2.0, float(status)]],
            dtype=torch.float32,
            device=self.device,
        )
        return self.model(x).item()


class PGNNPhase2Inference:
    def __init__(self, checkpoint_path: str, device: str = "cpu", hidden_dim: int = 32):
        self.device = device
        self.model = PGNNPhase2(hidden_dim=hidden_dim).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_delta_f(
            self,
            delta_f_1: float,
            difficulty: float,
            automation: float,
            workload: float,
            workload_max: float = 50.0,
    ) -> float:
        workload_norm = workload / workload_max
        x = torch.tensor(
            [[delta_f_1, difficulty, automation, workload_norm]],
            dtype=torch.float32,
            device=self.device,
        )

        scale_raw = self.model(x)
        scale = 1.0 + 0.2 * torch.tanh(scale_raw)
        delta2 = delta_f_1 * scale.item()
        return delta2