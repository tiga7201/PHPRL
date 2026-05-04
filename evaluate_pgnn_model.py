import os
import math
import json
import random
from typing import Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.pgnn import PGNNPhase1, PGNNPhase2
from train_pgnn import compute_delta_f_phase1

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_phase1_ckpt_path() -> str:
    return os.path.join(get_project_root(), "checkpoints", "pgnn_phase1.pt")


def get_phase2_ckpt_path() -> str:
    return os.path.join(get_project_root(), "checkpoints", "pgnn_phase2.pt")


def load_phase1_model(hidden_dim: int = 32, device: str = "cpu") -> PGNNPhase1:
    model = PGNNPhase1(hidden_dim=hidden_dim).to(device)
    ckpt = torch.load(get_phase1_ckpt_path(), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_phase2_model(hidden_dim: int = 32, device: str = "cpu") -> PGNNPhase2:
    model = PGNNPhase2(hidden_dim=hidden_dim).to(device)
    ckpt = torch.load(get_phase2_ckpt_path(), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_status_seq(blocks):
    """
    blocks: list of (status, length)
    status = 1 means working
    status = 0 means resting
    example:
        [(1, 12), (0, 15), (1, 20), (0, 10)]
    """
    seq = []
    for status, length in blocks:
        seq.extend([int(status)] * int(length))
    return seq


def validate_same_horizon(workers):
    lengths = [len(w["status_seq"]) for w in workers]
    if len(set(lengths)) != 1:
        raise ValueError(f"All workers must have the same total horizon, got lengths={lengths}")


@torch.no_grad()
def predict_phase1_delta(
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


@torch.no_grad()
def build_delta2_from_phase2(
    phase2_model: PGNNPhase2,
    delta1: float,
    difficulty: float,
    automation: float,
    workload: float,
    fatigue: float,
    status: int,
    workload_max: float,
    device: str,
):
    workload_norm = workload / workload_max
    x = torch.tensor(
        [[delta1, difficulty, automation, workload_norm, fatigue, float(status)]],
        dtype=torch.float32,
        device=device,
    )

    r = phase2_model(x)
    scale = 1.0 + phase2_model.max_correction * r

    delta1_tensor = torch.tensor([delta1], dtype=torch.float32, device=device)
    fatigue_tensor = torch.tensor([fatigue], dtype=torch.float32, device=device)
    status_tensor = torch.tensor([float(status)], dtype=torch.float32, device=device)

    max_change = torch.where(status_tensor > 0.5, 1.0 - fatigue_tensor, fatigue_tensor)
    zero_tensor = torch.zeros_like(delta1_tensor)

    delta2 = delta1_tensor * scale
    delta2_work = torch.clamp(delta2, zero_tensor, max_change)
    delta2_rest = torch.clamp(delta2, -max_change, zero_tensor)
    delta2 = torch.where(status_tensor > 0.5, delta2_work, delta2_rest)

    return delta2.item(), r.item(), scale.item()


def evaluate_phase1_fitting(
    phase1_model: PGNNPhase1,
    num_samples: int = 5000,
    device: str = "cpu",
) -> Dict:
    y_true = []
    y_pred = []

    for _ in range(num_samples):
        fatigue = random.random()
        f_p = random.choice([1, 2, 3])
        status = random.choice([0, 1])

        gt = compute_delta_f_phase1(fatigue, f_p, status)
        pred = predict_phase1_delta(
            phase1_model,
            fatigue=fatigue,
            f_p=f_p,
            status=status,
            device=device,
        )

        y_true.append(gt)
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }


def simulate_curve_exponential(
    f_p: int,
    fatigue0: float,
    status_seq: List[int],
) -> List[float]:
    fatigue = fatigue0
    curve = [fatigue]

    for status in status_seq:
        delta = compute_delta_f_phase1(fatigue, f_p, status)
        fatigue = max(0.0, min(1.0, fatigue + delta))
        curve.append(fatigue)

    return curve


def simulate_curve_phase1(
    phase1_model: PGNNPhase1,
    f_p: int,
    fatigue0: float,
    status_seq: List[int],
    device: str,
) -> List[float]:
    fatigue = fatigue0
    curve = [fatigue]

    for status in status_seq:
        delta = predict_phase1_delta(
            phase1_model,
            fatigue=fatigue,
            f_p=f_p,
            status=status,
            device=device,
        )
        fatigue = max(0.0, min(1.0, fatigue + delta))
        curve.append(fatigue)

    return curve


def simulate_curve_phase12(
    phase1_model: PGNNPhase1,
    phase2_model: PGNNPhase2,
    f_p: int,
    fatigue0: float,
    status_seq: List[int],
    difficulty_seq: List[float],
    automation_seq: List[float],
    workload_seq: List[float],
    workload_max: float,
    device: str,
) -> Dict:
    fatigue = fatigue0
    curve = [fatigue]
    r_list = []
    scale_list = []

    for t, status in enumerate(status_seq):
        delta1 = predict_phase1_delta(
            phase1_model,
            fatigue=fatigue,
            f_p=f_p,
            status=status,
            device=device,
        )
        delta2, r, scale = build_delta2_from_phase2(
            phase2_model=phase2_model,
            delta1=delta1,
            difficulty=difficulty_seq[t],
            automation=automation_seq[t],
            workload=workload_seq[t],
            fatigue=fatigue,
            status=status,
            workload_max=workload_max,
            device=device,
        )
        fatigue = max(0.0, min(1.0, fatigue + delta2))
        curve.append(fatigue)
        r_list.append(r)
        scale_list.append(scale)

    return {
        "curve": curve,
        "r_list": r_list,
        "scale_list": scale_list,
    }


def plot_phase1_curve_comparison(
    phase1_model: PGNNPhase1,
    device: str,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    workers = [
        {"f_p": 1, "fatigue0": 0.2},
        {"f_p": 2, "fatigue0": 0.2},
        {"f_p": 3, "fatigue0": 0.2},
    ]

    status_seq = [1] * 25 + [0] * 20 + [1] * 25 + [0] * 20
    time_steps = list(range(len(status_seq) + 1))

    plt.figure(figsize=(10, 6))

    for i, worker in enumerate(workers):
        exp_curve = simulate_curve_exponential(
            f_p=worker["f_p"],
            fatigue0=worker["fatigue0"],
            status_seq=status_seq,
        )
        p1_curve = simulate_curve_phase1(
            phase1_model=phase1_model,
            f_p=worker["f_p"],
            fatigue0=worker["fatigue0"],
            status_seq=status_seq,
            device=device,
        )

        plt.plot(time_steps, exp_curve, linestyle="-", label=f"Exp model, fp={worker['f_p']}")
        plt.plot(time_steps, p1_curve, linestyle="--", label=f"PGNN Phase 1, fp={worker['f_p']}")

    plt.xlabel("Time step")
    plt.ylabel("Fatigue level")
    plt.title("Phase 1 PGNN vs Exponential Fatigue Model")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = os.path.join(save_dir, "phase1_curve_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def plot_phase1_curve_comparison_general(
    phase1_model,
    device: str,
    save_path: str,
    workers,
    title: str,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    validate_same_horizon(workers)

    plt.figure(figsize=(18, 15))

    # 你可以自己换颜色
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    for idx, worker in enumerate(workers):
        name = worker.get("name", f"worker_{idx+1}")
        status_seq = worker["status_seq"]
        time_steps = list(range(len(status_seq) + 1))

        exp_curve = simulate_curve_exponential(
            f_p=worker["f_p"],
            fatigue0=worker["fatigue0"],
            status_seq=status_seq,
        )
        p1_curve = simulate_curve_phase1(
            phase1_model=phase1_model,
            f_p=worker["f_p"],
            fatigue0=worker["fatigue0"],
            status_seq=status_seq,
            device=device,
        )

        color = colors[idx % len(colors)]

        # 1) 指数模型：实线
        plt.plot(
            time_steps,
            exp_curve,
            color=color,
            linestyle="-",
            linewidth=2.2,
            # label=f"Exp, $W_{idx+1}$,  $f_{idx+1}^{{\\mathrm{{p}}}}={worker['f_p']}$"
            label = f"Exp,  $f_{idx + 1}^{{\\mathrm{{p}}}}={worker['f_p']}$"
        )

        # 2) PGNN：空心marker
        # 如果点太密，可以隔几个点画一个
        marker_idx = list(range(0, len(time_steps), 2))
        if marker_idx[-1] != len(time_steps) - 1:
            marker_idx.append(len(time_steps) - 1)

        plt.plot(
            [time_steps[i] for i in marker_idx],
            [p1_curve[i] for i in marker_idx],
            color=color,
            linestyle="None",
            marker="o",
            markersize=14,
            markerfacecolor="none",
            markeredgewidth=1.6,
            # label=f"PGNN, $W_{idx+1}$,  $f_{worker['f_p']}^{{\\mathrm{{p}}}}={worker['f_p']}$"
            label = f"PGNN,  $f_{idx + 1}^{{\\mathrm{{p}}}}={worker['f_p']}$"
        )

    plt.xlabel("Time", fontsize=48, labelpad=15)
    plt.ylabel("Fatigue level", fontsize=48, labelpad=15)
    # plt.title(title, fontsize=16)
    plt.xticks(fontsize=48)
    plt.yticks(fontsize=48)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    plt.legend(fontsize=40, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def plot_phase1_shared_status_comparison(
    phase1_model,
    device: str,
    save_dir: str,
):
    """
    All workers share the same status sequence.
    """
    os.makedirs(save_dir, exist_ok=True)

    shared_status_seq = build_status_seq([
        (1, 12),
        (0, 30),
        (1, 20),
        (0, 30),
        (1, 20),
        (0, 18),
    ])

    workers = [
        {
            "name": "worker_1",
            "f_p": 1,
            "fatigue0": 0.0,
            "status_seq": shared_status_seq,
        },
        {
            "name": "worker_2",
            "f_p": 2,
            "fatigue0": 0.0,
            "status_seq": shared_status_seq,
        },
        {
            "name": "worker_3",
            "f_p": 3,
            "fatigue0": 0.0,
            "status_seq": shared_status_seq,
        },
    ]

    save_path = os.path.join(save_dir, "phase1_curve_shared_status.png")

    return plot_phase1_curve_comparison_general(
        phase1_model=phase1_model,
        device=device,
        save_path=save_path,
        workers=workers,
        title="Phase 1 PGNN vs Exponential Model (Shared Status Sequence)",
    )

def plot_phase1_individual_status_comparison(
    phase1_model,
    device: str,
    save_dir: str,
):
    """
    Each worker has its own status sequence,
    but all sequences share the same total horizon.
    """
    os.makedirs(save_dir, exist_ok=True)

    workers = [
        {
            "name": "worker_1",
            "f_p": 1,
            "fatigue0": 0.0,
            "status_seq": build_status_seq([
                (1, 10),
                (0, 30),
                (1, 10),
                (0, 20),
                (1, 25),
                (0, 30),
            ]),  # total = 125
        },
        {
            "name": "worker_2",
            "f_p": 2,
            "fatigue0": 0.1,
            "status_seq": build_status_seq([
                (1, 10),
                (0, 20),
                (1, 15),
                (0, 25),
                (1, 25),
                (0, 30),
            ]),  # total = 125
        },
        {
            "name": "worker_3",
            "f_p": 3,
            "fatigue0": 0.2,
            "status_seq": build_status_seq([
                (1, 25),
                (0, 15),
                (1, 20),
                (0, 10),
                (1, 30),
                (0, 25),
            ]),  # total = 125
        },
    ]

    save_path = os.path.join(save_dir, "phase1_curve_individual_status.png")

    return plot_phase1_curve_comparison_general(
        phase1_model=phase1_model,
        device=device,
        save_path=save_path,
        workers=workers,
        title="Phase 1 PGNN vs Exponential Model (Individual Status Sequences)",
    )

def evaluate_phase2_monotonicity(
    phase1_model: PGNNPhase1,
    phase2_model: PGNNPhase2,
    device: str,
    workload_max: float = 50.0,
    save_dir: str = "eval_results/pgnn_eval",
):
    os.makedirs(save_dir, exist_ok=True)

    fatigue = 0.4
    f_p = 2
    status = 1
    delta1 = predict_phase1_delta(phase1_model, fatigue, f_p, status, device)

    grids = np.linspace(0.1, 0.9, 21)

    monotonic_results = {}

    settings = [
        ("workload", {"difficulty": 1.0, "automation": 1.0}),
        ("difficulty", {"workload": 20.0, "automation": 1.0}),
        ("automation", {"workload": 20.0, "difficulty": 1.0}),
    ]

    plt.figure(figsize=(12, 4))

    for idx, (target, fixed) in enumerate(settings, start=1):
        values = []
        delta2_list = []

        for g in grids:
            if target == "workload":
                workload = float(g * workload_max)
                difficulty = fixed["difficulty"]
                automation = fixed["automation"]
                x_val = workload
            elif target == "difficulty":
                workload = fixed["workload"]
                difficulty = float(0.8 + 0.4 * g)
                automation = fixed["automation"]
                x_val = difficulty
            else:
                workload = fixed["workload"]
                difficulty = fixed["difficulty"]
                automation = float(0.8 + 0.4 * g)
                x_val = automation

            delta2, r, scale = build_delta2_from_phase2(
                phase2_model=phase2_model,
                delta1=delta1,
                difficulty=difficulty,
                automation=automation,
                workload=workload,
                fatigue=fatigue,
                status=status,
                workload_max=workload_max,
                device=device,
            )

            values.append(x_val)
            delta2_list.append(delta2)

        values = np.array(values)
        delta2_list = np.array(delta2_list)

        grad = np.gradient(delta2_list, values)
        grad_mean = float(np.mean(grad))

        if target in ["workload", "difficulty"]:
            trend_ok = bool(grad_mean > 0)
        else:
            trend_ok = bool(grad_mean < 0)

        monotonic_results[target] = {
            "gradient_mean": grad_mean,
            "trend_ok": trend_ok,
            "delta2_min": float(np.min(delta2_list)),
            "delta2_max": float(np.max(delta2_list)),
        }

        plt.subplot(1, 3, idx)
        plt.plot(values, delta2_list, marker="o")
        plt.axhline(delta1, linestyle="--")
        plt.title(f"{target}, trend_ok={trend_ok}")
        plt.xlabel(target)
        plt.ylabel("delta_f_2")
        plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, "phase2_monotonicity.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return monotonic_results, save_path


def build_workload_seq_from_status(
    status_seq: List[int],
    start_workload: float = 0.0,
    work_increment: float = 1.0,
) -> List[float]:
    """
    Build workload sequence from status sequence.
    Workload increases only during working periods and stays unchanged during resting periods.
    """
    workload_seq = []
    current_workload = float(start_workload)

    for status in status_seq:
        workload_seq.append(current_workload)
        if status == 1:
            current_workload += work_increment

    return workload_seq


def plot_phase12_curve_comparison(
    phase1_model,
    phase2_model,
    device: str,
    workload_max: float,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    # same worker setting
    f_p = 2
    fatigue0 = 0.0

    # same status sequence for all scenarios
    status_seq = build_status_seq([
        (1, 18),
        (0, 15),
        (1, 12),
        (0, 25),
        (1, 30),
        (0, 25),
    ])

    # baseline exponential curve
    exp_curve = simulate_curve_exponential(
        f_p=f_p,
        fatigue0=fatigue0,
        status_seq=status_seq,
    )

    # -------- three dynamic-condition scenarios --------
    T = len(status_seq)
    workload_seq = build_workload_seq_from_status(
        status_seq=status_seq,
        start_workload=0.0,
        work_increment=1.0,
    )

    scenarios = [
        {
            "name": r'$f_{ij}^{\mathrm{d}}=0.8,\ f_{k}^{\mathrm{a}}=1.2$',
            "difficulty_seq": [0.8] * T,
            "automation_seq": [1.2] * T,
            "workload_seq": workload_seq,
            "color": "#1f77b4",
        },
        {
            "name": r'$f_{ij}^{\mathrm{d}}=1.0,\ f_{k}^{\mathrm{a}}=1.0$',
            "difficulty_seq": [1.0] * T,
            "automation_seq": [1.0] * T,
            "workload_seq": workload_seq,
            "color": "#ff7f0e",
        },
        {
            "name": r'$f_{ij}^{\mathrm{d}}=1.2,\ f_{k}^{\mathrm{a}}=0.8$',
            "difficulty_seq": [1.2] * T,
            "automation_seq": [0.8] * T,
            "workload_seq": workload_seq,
            "color": "#d62728",
        },
    ]

    time_steps = list(range(T + 1))

    plt.figure(figsize=(18, 12))

    # baseline: exponential model
    plt.plot(
        time_steps,
        exp_curve,
        color="black",
        linestyle="-",
        linewidth=2.4,
        label="Exponential fatigue model",
    )

    # phase1+2 curves under different dynamic conditions
    for sc in scenarios:
        result = simulate_curve_phase12(
            phase1_model=phase1_model,
            phase2_model=phase2_model,
            f_p=f_p,
            fatigue0=fatigue0,
            status_seq=status_seq,
            difficulty_seq=sc["difficulty_seq"],
            automation_seq=sc["automation_seq"],
            workload_seq=sc["workload_seq"],
            workload_max=workload_max,
            device=device,
        )

        plt.plot(
            time_steps,
            result["curve"],
            color=sc["color"],
            linestyle="--",
            linewidth=2.0,
            label=sc["name"],
        )

    plt.xlabel("Time", fontsize=30, labelpad=15)
    plt.ylabel("Fatigue level", fontsize=30, labelpad=15)
    # plt.title("Fatigue curves under different dynamic conditions", fontsize=14)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    plt.legend(fontsize=30, loc="best")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "phase12_curve_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = os.path.join("eval_results", "pgnn_eval")
    os.makedirs(save_dir, exist_ok=True)

    phase1_model = load_phase1_model(hidden_dim=32, device=device)
    phase2_model = load_phase2_model(hidden_dim=32, device=device)

    phase1_fit_result = evaluate_phase1_fitting(
        phase1_model=phase1_model,
        num_samples=5000,
        device=device,
    )

    phase1_curve_shared_path = plot_phase1_shared_status_comparison(
        phase1_model=phase1_model,
        device=device,
        save_dir=save_dir,
    )

    phase1_curve_individual_path = plot_phase1_individual_status_comparison(
        phase1_model=phase1_model,
        device=device,
        save_dir=save_dir,
    )

    monotonic_result, monotonic_fig_path = evaluate_phase2_monotonicity(
        phase1_model=phase1_model,
        phase2_model=phase2_model,
        device=device,
        workload_max=50.0,
        save_dir=save_dir,
    )

    phase12_curve_path = plot_phase12_curve_comparison(
        phase1_model=phase1_model,
        phase2_model=phase2_model,
        device=device,
        workload_max=50.0,
        save_dir=save_dir,
    )

    summary = {
        "phase1_fitting": phase1_fit_result,
        "phase2_monotonicity": monotonic_result,
        "figures": {
            "phase1_curve_shared_status": phase1_curve_shared_path,
            "phase1_curve_individual_status": phase1_curve_individual_path,
            "phase2_monotonicity": monotonic_fig_path,
            "phase12_curve_comparison": phase12_curve_path,
        },
    }

    save_json = os.path.join(save_dir, "pgnn_evaluation_summary.json")
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Phase 1 fitting:")
    print(summary["phase1_fitting"])
    print("\nPhase 2 monotonicity:")
    print(summary["phase2_monotonicity"])
    print(f"\nSaved summary to: {save_json}")


if __name__ == "__main__":
    main()