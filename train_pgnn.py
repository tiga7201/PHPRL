import os
import math
import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models.pgnn import PGNNPhase1


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def get_lambda_mu_by_physical_condition(f_p: int) -> Tuple[float, float]:
    """
    f_p in {1,2,3}
    1 -> low physical condition
    2 -> medium
    3 -> high
    """
    if f_p == 1:
        return 0.05, 0.01
    elif f_p == 2:
        return 0.03, 0.03
    elif f_p == 3:
        return 0.01, 0.05
    else:
        raise ValueError(f"Invalid physical condition: {f_p}")


def compute_delta_f_phase1(fatigue: float, f_p: int, status: int) -> float:
    """
    status = 1: working
    status = 0: resting
    """
    lam, mu = get_lambda_mu_by_physical_condition(f_p)

    if status == 1:
        delta_f = (1.0 - fatigue) * (1.0 - math.exp(-lam))
    else:
        delta_f = -fatigue * (1.0 - math.exp(-mu))

    return delta_f


def generate_phase1_dataset(num_samples: int = 50000) -> TensorDataset:
    xs = []
    ys = []

    for _ in range(num_samples):
        fatigue = random.random()          # in [0,1]
        f_p = random.choice([1, 2, 3])     # physical condition
        status = random.choice([0, 1])     # 0 resting, 1 working

        delta_f = compute_delta_f_phase1(fatigue, f_p, status)

        # simple normalization:
        # fatigue in [0,1]
        # f_p in {1,2,3} -> map to [0,1]
        # status in {0,1}
        x = [
            fatigue,
            (f_p - 1) / 2.0,
            float(status),
        ]
        y = [delta_f]

        xs.append(x)
        ys.append(y)

    x_tensor = torch.tensor(xs, dtype=torch.float32)
    y_tensor = torch.tensor(ys, dtype=torch.float32).squeeze(-1)
    return TensorDataset(x_tensor, y_tensor)


def train_phase1(
    hidden_dim: int = 32,
    lr: float = 1e-3,
    batch_size: int = 256,
    num_epochs: int = 50,
    num_samples: int = 50000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    train_dataset = generate_phase1_dataset(num_samples=num_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = PGNNPhase1(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = []

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_count += bs

        avg_loss = total_loss / max(total_count, 1)
        history.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:03d} | phase1_loss = {avg_loss:.8f}")

    return {
        "model": model,
        "history": history,
    }


@torch.no_grad()
def quick_eval_phase1(model: PGNNPhase1, device: str):
    model.eval()

    test_cases = [
        (0.2, 1, 1),
        (0.2, 2, 1),
        (0.2, 3, 1),
        (0.8, 1, 0),
        (0.8, 2, 0),
        (0.8, 3, 0),
    ]

    print("\nQuick evaluation:")
    for fatigue, f_p, status in test_cases:
        gt = compute_delta_f_phase1(fatigue, f_p, status)

        x = torch.tensor(
            [[fatigue, (f_p - 1) / 2.0, float(status)]],
            dtype=torch.float32,
            device=device,
        )
        pred = model(x).item()

        print(
            f"fatigue={fatigue:.2f}, f_p={f_p}, status={status} | "
            f"gt={gt:.6f}, pred={pred:.6f}, abs_err={abs(gt - pred):.6e}"
        )


def save_phase1_model(model: PGNNPhase1, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "PGNNPhase1",
        },
        path,
    )


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    result = train_phase1(
        hidden_dim=32,
        lr=1e-3,
        batch_size=256,
        num_epochs=50,
        num_samples=50000,
        device=device,
    )

    model = result["model"]
    quick_eval_phase1(model, device=device)

    save_path = "checkpoints/pgnn_phase1.pt"
    save_phase1_model(model, save_path)
    print(f"\nSaved phase-1 PGNN to: {save_path}")


if __name__ == "__main__":
    main()