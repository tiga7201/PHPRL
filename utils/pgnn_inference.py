from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from models.pgnn import PGNNPhase1, PGNNPhase2


class _NumpyMLP:
    """Low-overhead CPU inference for the tiny PGNN MLPs."""

    def __init__(self, model: nn.Module):
        linear_layers = [
            layer for layer in model.net
            if isinstance(layer, nn.Linear)
        ]
        self.weights = [
            layer.weight.detach().cpu().numpy().copy()
            for layer in linear_layers
        ]
        self.biases = [
            layer.bias.detach().cpu().numpy().copy()
            for layer in linear_layers
        ]

    def predict(self, values: Sequence[float]) -> float:
        x = np.asarray(values, dtype=np.float32)
        last_layer = len(self.weights) - 1

        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = weight @ x + bias
            if index != last_layer:
                np.maximum(x, 0.0, out=x)

        return float(x[0])


class PGNNPhase1Inference:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        hidden_dim: int = 32,
        use_fast_cpu: bool = True,
    ):
        self.device = device
        self.model = PGNNPhase1(hidden_dim=hidden_dim).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self._numpy_model = (
            _NumpyMLP(self.model)
            if use_fast_cpu and torch.device(device).type == "cpu"
            else None
        )

    @torch.inference_mode()
    def predict_delta_f(self, fatigue: float, f_p: int, status: int) -> float:
        values = [fatigue, (f_p - 1) / 2.0, float(status)]
        if self._numpy_model is not None:
            return self._numpy_model.predict(values)

        x = torch.tensor(
            [values],
            dtype=torch.float32,
            device=self.device,
        )
        return self.model(x).item()


class PGNNPhase2Inference:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        hidden_dim: int = 32,
        use_fast_cpu: bool = True,
    ):
        self.device = device
        self.model = PGNNPhase2(hidden_dim=hidden_dim).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self._numpy_model = (
            _NumpyMLP(self.model)
            if use_fast_cpu and torch.device(device).type == "cpu"
            else None
        )
        self.max_correction = float(self.model.max_correction)

    @torch.inference_mode()
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
        values = [
            delta_f_1,
            difficulty,
            automation,
            workload_norm,
            fatigue,
            float(status),
        ]

        if self._numpy_model is not None:
            raw = self._numpy_model.predict(values)
            correction = float(np.tanh(raw))
            delta_f_2 = np.float32(delta_f_1) * np.float32(
                1.0 + self.max_correction * correction
            )
            max_change = np.float32(
                1.0 - fatigue if status > 0.5 else fatigue
            )
            if status > 0.5:
                return max(0.0, min(delta_f_2, max_change))
            return max(-max_change, min(delta_f_2, 0.0))

        x = torch.tensor(
            [values],
            dtype=torch.float32,
            device=self.device,
        )

        r = self.model(x)
        scale = 1.0 + self.max_correction * r

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
