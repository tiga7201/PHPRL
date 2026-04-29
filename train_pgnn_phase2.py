import os
import random
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.pgnn import PGNNPhase1, PGNNPhase2


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def load_phase1_model(
    path: str,
    hidden_dim: int = 32,
    device: str = "cpu",
) -> PGNNPhase1:
    model = PGNNPhase1(hidden_dim=hidden_dim).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def phase1_predict(
    model: PGNNPhase1,
    fatigue: float,
    f_p: int,
    status: int,
    device: str,
) -> float:
    x = torch.tensor(
        [[fatigue, (f_p - 1) / 2.0, float(status)]],
        dtype=torch.float32,
        device=device,
    )
    return model(x).item()


def generate_phase2_dataset(
    phase1_model: PGNNPhase1,
    num_samples: int = 50000,
    workload_max: float = 50.0,
    device: str = "cpu",
):
    xs = []

    diff_list = []
    auto_list = []
    work_list = []
    delta1_list = []
    fatigue_list = []
    status_list = []

    for _ in range(num_samples):
        fatigue = random.random()
        f_p = random.choice([1, 2, 3])
        status = random.choice([0, 1])

        difficulty = random.choice([0.8, 1.0, 1.2])
        automation = random.choice([0.8, 1.0, 1.2])
        workload = random.uniform(0.0, workload_max)

        delta_f_1 = phase1_predict(
            phase1_model,
            fatigue=fatigue,
            f_p=f_p,
            status=status,
            device=device,
        )

        workload_norm = workload / workload_max

        x = [
            delta_f_1,
            difficulty,
            automation,
            workload_norm,
            fatigue,
            float(status),
        ]
        xs.append(x)

        diff_list.append(difficulty)
        auto_list.append(automation)
        work_list.append(workload_norm)
        delta1_list.append(delta_f_1)
        fatigue_list.append(fatigue)
        status_list.append(status)

    x_tensor = torch.tensor(xs, dtype=torch.float32)
    diff_tensor = torch.tensor(diff_list, dtype=torch.float32)
    auto_tensor = torch.tensor(auto_list, dtype=torch.float32)
    work_tensor = torch.tensor(work_list, dtype=torch.float32)
    delta1_tensor = torch.tensor(delta1_list, dtype=torch.float32)
    fatigue_tensor = torch.tensor(fatigue_list, dtype=torch.float32)
    status_tensor = torch.tensor(status_list, dtype=torch.float32)

    return {
        "dataset": TensorDataset(
            x_tensor,
            diff_tensor,
            auto_tensor,
            work_tensor,
            delta1_tensor,
            fatigue_tensor,
            status_tensor,
        )
    }

def build_delta2_from_r(
    r: torch.Tensor,
    delta1: torch.Tensor,
    fatigue: torch.Tensor,
    status: torch.Tensor,
    max_correction: float = 0.5,
):
    """
    delta2 = delta1 * (1 + max_correction * r)
    then clamp by physical bounds
    """
    scale = 1.0 + max_correction * r
    delta2 = delta1 * scale

    max_change = torch.where(status > 0.5, 1.0 - fatigue, fatigue)
    zero_tensor = torch.zeros_like(delta2)
    work_mask = status > 0.5

    delta2_work = torch.clamp(delta2, zero_tensor, max_change)
    delta2_rest = torch.clamp(delta2, -max_change, zero_tensor)

    delta2 = torch.where(work_mask, delta2_work, delta2_rest)
    return delta2, scale

def compute_phase2_loss(
    model: PGNNPhase2,
    x_batch: torch.Tensor,
    diff_batch: torch.Tensor,
    auto_batch: torch.Tensor,
    work_batch: torch.Tensor,
    delta1_batch: torch.Tensor,
    fatigue_batch: torch.Tensor,
    status_batch: torch.Tensor,
    xi1: float = 0.3,
    xi2: float = 1.0,
    xi3: float = 0.8,
    xi4: float = 0.5,
):
    """
    xi1: data/basic closeness
    xi2: monotonicity on relative correction
    xi3: reasonableness
    xi4: encourage non-trivial correction magnitude
    """
    relu = nn.ReLU()

    r = model(x_batch)
    delta2, scale = build_delta2_from_r(
        r=r,
        delta1=delta1_batch,
        fatigue=fatigue_batch,
        status=status_batch,
        max_correction=model.max_correction,
    )

    # 1) data/basic loss: stay reasonably close to delta1
    loss_ba = ((delta2 - delta1_batch) ** 2).mean()

    # 2) monotonicity on relative correction
    # relative correction = (delta2 - delta1) / |delta1|
    rel_corr = (delta2 - delta1_batch) / (torch.abs(delta1_batch) + 1e-6)

    working_mask = (status_batch > 0.5)
    if working_mask.sum() >= 2:
        rc = rel_corr[working_mask]
        diffw = diff_batch[working_mask]
        autow = auto_batch[working_mask]
        workw = work_batch[working_mask]
        rw = r[working_mask]

        rc_diff = rc.unsqueeze(1) - rc.unsqueeze(0)
        r_diff = rw.unsqueeze(1) - rw.unsqueeze(0)

        diff_diff = diffw.unsqueeze(1) - diffw.unsqueeze(0)
        auto_diff = autow.unsqueeze(1) - autow.unsqueeze(0)
        work_diff = workw.unsqueeze(1) - workw.unsqueeze(0)

        valid_diff = (torch.abs(diff_diff) > 1e-6).float()
        valid_auto = (torch.abs(auto_diff) > 1e-6).float()
        valid_work = (torch.abs(work_diff) > 1e-6).float()

        # difficulty ↑ -> relative correction ↑
        penalty_diff = torch.sigmoid(-10.0 * rc_diff * torch.sign(diff_diff)) * valid_diff
        penalty_diff_r = torch.sigmoid(-10.0 * r_diff * torch.sign(diff_diff)) * valid_diff

        # automation ↑ -> relative correction ↓
        penalty_auto = torch.sigmoid(10.0 * rc_diff * torch.sign(auto_diff)) * valid_auto
        penalty_auto_r = torch.sigmoid(10.0 * r_diff * torch.sign(auto_diff)) * valid_auto

        # workload ↑ -> relative correction ↑
        penalty_work = torch.sigmoid(-10.0 * rc_diff * torch.sign(work_diff)) * valid_work
        penalty_work_r = torch.sigmoid(-10.0 * r_diff * torch.sign(work_diff)) * valid_work

        loss_mo = (
            penalty_diff.mean() + 0.5 * penalty_diff_r.mean() +
            penalty_auto.mean() + 0.5 * penalty_auto_r.mean() +
            penalty_work.mean() + 0.5 * penalty_work_r.mean()
        )
    else:
        loss_mo = torch.zeros(1, device=x_batch.device).mean()

    # 3) reasonableness
    rel_correction_abs = torch.abs(rel_corr)

    # too large > 0.8 or too small < 0.05
    loss_reason = (
        relu(rel_correction_abs - 0.8) +
        relu(0.05 - rel_correction_abs)
    ).mean()

    # 4) encourage r not to collapse to zero
    avg_abs_r = torch.mean(torch.abs(r))
    loss_mag = relu(0.1 - avg_abs_r)

    total_loss = (
        xi1 * loss_ba +
        xi2 * loss_mo +
        xi3 * loss_reason +
        xi4 * loss_mag
    )

    return total_loss, {
        "loss_total": float(total_loss.item()),
        "loss_ba": float(loss_ba.item()),
        "loss_mo": float(loss_mo.item()),
        "loss_reason": float(loss_reason.item()),
        "loss_mag": float(loss_mag.item()),
        "avg_abs_r": float(avg_abs_r.item()),
    }


def train_phase2(
    phase1_ckpt_path: str,
    hidden_dim: int = 32,
    lr: float = 1e-3,
    batch_size: int = 256,
    num_epochs: int = 50,
    num_samples: int = 50000,
    workload_max: float = 50.0,
    xi1: float = 0.2,
    xi2: float = 0.6,
    xi3: float = 0.2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    phase1_model = load_phase1_model(
        path=phase1_ckpt_path,
        hidden_dim=hidden_dim,
        device=device,
    )

    data = generate_phase2_dataset(
        phase1_model=phase1_model,
        num_samples=num_samples,
        workload_max=workload_max,
        device=device,
    )

    loader = DataLoader(data["dataset"], batch_size=batch_size, shuffle=True)

    model = PGNNPhase2(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        total_ba = 0.0
        total_mo = 0.0
        total_in = 0.0

        for batch in loader:
            x_batch, diff_batch, auto_batch, work_batch, delta1_batch, fatigue_batch, status_batch = batch

            x_batch = x_batch.to(device)
            diff_batch = diff_batch.to(device)
            auto_batch = auto_batch.to(device)
            work_batch = work_batch.to(device)
            delta1_batch = delta1_batch.to(device)
            status_batch = status_batch.to(device)

            loss, stats = compute_phase2_loss(
                model=model,
                x_batch=x_batch,
                diff_batch=diff_batch,
                auto_batch=auto_batch,
                work_batch=work_batch,
                delta1_batch=delta1_batch,
                fatigue_batch=fatigue_batch,
                status_batch=status_batch,
                xi1=0.3,
                xi2=1.0,
                xi3=0.8,
                xi4=0.5,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x_batch.size(0)
            total_loss += loss.item() * bs
            total_ba += stats["loss_ba"] * bs
            total_mo += stats["loss_mo"] * bs
            total_in += stats["loss_in"] * bs
            total_count += bs

        avg_loss = total_loss / total_count
        avg_ba = total_ba / total_count
        avg_mo = total_mo / total_count
        avg_in = total_in / total_count

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "loss_ba": avg_ba,
            "loss_mo": avg_mo,
            "loss_in": avg_in,
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"loss={avg_loss:.8f} | "
                f"ba={avg_ba:.8f} | "
                f"mo={avg_mo:.8f} | "
                f"reason={avg_reason:.8f} | "
                f"mag={avg_mag:.8f} | "
                f"|r|={avg_abs_r:.4f}"
            )

    return {
        "model": model,
        "history": history,
    }


@torch.no_grad()
def quick_eval_phase2(
    phase1_ckpt_path: str,
    phase2_model: PGNNPhase2,
    hidden_dim: int,
    device: str,
):
    phase1_model = load_phase1_model(
        path=phase1_ckpt_path,
        hidden_dim=hidden_dim,
        device=device,
    )

    test_cases = [
        # (fatigue, f_p, status, difficulty, automation, workload)
        (0.3, 2, 1, 0.8, 1.2, 5.0),
        (0.3, 2, 1, 1.2, 0.8, 30.0),
        (0.7, 1, 0, 1.0, 1.0, 10.0),
    ]

    print("\nQuick evaluation Phase-2:")
    for fatigue, f_p, status, difficulty, automation, workload in test_cases:
        delta1 = phase1_predict(phase1_model, fatigue, f_p, status, device)
        workload_norm = workload / 50.0

        x = torch.tensor(
            [[delta1, difficulty, automation, workload_norm, fatigue, float(status)]],
            dtype=torch.float32,
            device=device,
        )

        r = phase2_model(x)
        delta1_tensor = torch.tensor([delta1], dtype=torch.float32, device=device)
        fatigue_tensor = torch.tensor([fatigue], dtype=torch.float32, device=device)
        status_tensor = torch.tensor([float(status)], dtype=torch.float32, device=device)

        delta2, scale = build_delta2_from_r(
            r=r,
            delta1=delta1_tensor,
            fatigue=fatigue_tensor,
            status=status_tensor,
            max_correction=phase2_model.max_correction,
        )

        print(
            f"fatigue={fatigue:.2f}, f_p={f_p}, status={status}, "
            f"d={difficulty:.1f}, a={automation:.1f}, w={workload:.1f} | "
            f"delta1={delta1:.6f}, delta2={delta2.item():.6f}, "
            f"r={r.item():.6f}, scale={scale.item():.6f}"
        )


def save_phase2_model(model: PGNNPhase2, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "PGNNPhase2",
        },
        path,
    )


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    phase1_ckpt_path = "checkpoints/pgnn_phase1.pt"

    result = train_phase2(
        phase1_ckpt_path=phase1_ckpt_path,
        hidden_dim=32,
        lr=1e-3,
        batch_size=256,
        num_epochs=50,
        num_samples=50000,
        workload_max=50.0,
        xi1=0.2,
        xi2=0.6,
        xi3=0.2,
        device=device,
    )

    model = result["model"]
    quick_eval_phase2(
        phase1_ckpt_path=phase1_ckpt_path,
        phase2_model=model,
        hidden_dim=32,
        device=device,
    )

    save_path = "checkpoints/pgnn_phase2.pt"
    save_phase2_model(model, save_path)
    print(f"\nSaved phase-2 PGNN to: {save_path}")


if __name__ == "__main__":
    main()