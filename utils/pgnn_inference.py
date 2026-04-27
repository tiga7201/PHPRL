import torch

from models.pgnn import PGNNPhase1


class PGNNPhase1Inference:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.model = PGNNPhase1(hidden_dim=32).to(device)

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
        delta_f = self.model(x).item()
        return delta_f
