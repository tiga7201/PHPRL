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
        fatigue: float,
        status: int,
        workload_max: float = 50.0,
    ) -> float:
        workload_norm = workload / workload_max
        x = torch.tensor(
            [[delta_f_1, difficulty, automation, workload_norm, fatigue, float(status)]],
            dtype=torch.float32,
            device=self.device,
        )

        r = self.model(x)
        scale = 1.0 + self.model.max_correction * r

        delta1_tensor = torch.tensor([delta_f_1], dtype=torch.float32, device=self.device)
        fatigue_tensor = torch.tensor([fatigue], dtype=torch.float32, device=self.device)
        status_tensor = torch.tensor([float(status)], dtype=torch.float32, device=self.device)

        max_change = torch.where(status_tensor > 0.5, 1.0 - fatigue_tensor, fatigue_tensor)
        zero_tensor = torch.zeros_like(delta1_tensor)

        delta2 = delta1_tensor * scale
        delta2_work = torch.clamp(delta2, zero_tensor, max_change)
        delta2_rest = torch.clamp(delta2, -max_change, zero_tensor)
        delta2 = torch.where(status_tensor > 0.5, delta2_work, delta2_rest)

        return delta2.item()