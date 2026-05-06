import os
import json
from typing import Dict, List, Any, Tuple

import torch
import matplotlib.pyplot as plt

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state

from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from rl.sac_agent import SACAgent
from rl.pdr_baselines import select_pdr_action


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_pgnn_phase1_path() -> str:
    return os.path.join(get_project_root(), "checkpoints", "pgnn_phase1.pt")


def get_pgnn_phase2_path() -> str:
    return os.path.join(get_project_root(), "checkpoints", "pgnn_phase2.pt")


def make_env(
    seed: int,
    num_jobs: int,
    num_machines: int,
    num_workers: int,
    min_ops_per_job: int,
    max_ops_per_job: int,
    use_phase1: bool = True,
    use_phase2: bool = True,
) -> FJSPWFEnv:
    instance = generate_random_instance(
        seed=seed,
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_workers=num_workers,
        min_ops_per_job=min_ops_per_job,
        max_ops_per_job=max_ops_per_job,
    )

    env = FJSPWFEnv(instance)

    if use_phase1:
        env.load_pgnn_phase1(get_pgnn_phase1_path(), device="cpu")

    if use_phase2:
        phase2_path = get_pgnn_phase2_path()
        if os.path.exists(phase2_path):
            env.load_pgnn_phase2(phase2_path, device="cpu")

    return env


def load_actor_checkpoint(actor, checkpoint_path: str, map_location: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
        actor.load_state_dict(ckpt["actor_state_dict"])
    else:
        actor.load_state_dict(ckpt)


def build_rl_agent(actor_ckpt_path: str, hidden_dim: int = 64, num_layers: int = 2):
    actor = SHyperActorFull(hidden_dim=hidden_dim, num_layers=num_layers)
    q1 = SHyperQCriticFull(hidden_dim=hidden_dim, num_layers=num_layers)
    q2 = SHyperQCriticFull(hidden_dim=hidden_dim, num_layers=num_layers)

    agent = SACAgent(
        actor=actor,
        q1=q1,
        q2=q2,
        lr=1e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        device="cpu",
    )

    load_actor_checkpoint(actor, actor_ckpt_path, map_location="cpu")
    actor.eval()
    return agent


def finalize_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trace = sorted(trace, key=lambda x: x["time"])
    merged = []
    for item in trace:
        if merged and abs(merged[-1]["time"] - item["time"]) < 1e-9:
            merged[-1] = item
        else:
            merged.append(item)
    return merged


def run_case_rl(env: FJSPWFEnv, agent) -> Dict[str, Any]:
    env.reset()
    done = False
    last_info = None

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            env._advance_to_next_event()
            continue

        graph_state = build_hypergraph_state(env)
        decision = agent.actor.select_greedy_action(graph_state)
        action = decision["action"]
        _, _, done, info = env.step(action)
        last_info = info

    return {
        "makespan": float(last_info["makespan"]),
        "trace": finalize_trace(env.fatigue_time_trace),
        "final_schedule": [str(x) for x in env.schedule],
    }


def run_case_pdr(env: FJSPWFEnv, rule: str) -> Dict[str, Any]:
    env.reset()
    done = False
    last_info = None

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            env._advance_to_next_event()
            continue

        action = select_pdr_action(env, rule)
        _, _, done, info = env.step(action)
        last_info = info

    return {
        "makespan": float(last_info["makespan"]),
        "trace": finalize_trace(env.fatigue_time_trace),
        "final_schedule": [str(x) for x in env.schedule],
    }


def get_worker_curve(trace: List[Dict[str, Any]], worker_id: int) -> Tuple[List[float], List[float]]:
    times = [float(item["time"]) for item in trace]
    fatigue = [float(item["worker_fatigue"][worker_id]) for item in trace]
    return times, fatigue


def plot_worker_fatigue_curves(
    save_path: str,
    worker_id: int,
    method_results: Dict[str, Dict[str, Any]],
    method_display_names: Dict[str, str],
    method_colors: Dict[str, str],
    global_max_makespan: float,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(9, 5.5))

    for method_name, result in method_results.items():
        times, fatigue = get_worker_curve(result["trace"], worker_id)

        plt.plot(
            times,
            fatigue,
            color=method_colors.get(method_name, "black"),
            linewidth=1.4,
            alpha=0.95,
            label=method_display_names.get(method_name, method_name),
        )

    plt.xlim(0.0, global_max_makespan)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Time", fontsize=13, labelpad=10)
    plt.ylabel("Fatigue level", fontsize=13, labelpad=10)
    plt.title(f"Fatigue evolution of worker {worker_id}", fontsize=14)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    plt.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    case_config = {
        "seed": 100,
        "num_jobs": 10,
        "num_machines": 5,
        "num_workers": 3,
        "min_ops_per_job": 3,
        "max_ops_per_job": 7,
        "hidden_dim": 64,
        "num_layers": 2,
        "use_phase1": True,
        "use_phase2": True,
    }

    methods = {
        "pref_3stage_parallel_best": {
            "type": "rl",
            "actor_ckpt": "best_pref3stage_parallel_actor.pt",
            "display_name": "Proposed",
            "color": "#d62728",
        },
        "full_sac": {
            "type": "rl",
            "actor_ckpt": "best_fullsac_actor.pt",
            "display_name": "Full-SAC",
            "color": "#1f77b4",
        },
        "spt": {
            "type": "pdr",
            "rule": "SPT",
            "display_name": "SPT",
            "color": "#2ca02c",
        },
        "fifo": {
            "type": "pdr",
            "rule": "FIFO",
            "display_name": "FIFO",
            "color": "#ff7f0e",
        },
        "mwkr": {
            "type": "pdr",
            "rule": "MWKR",
            "display_name": "MWKR",
            "color": "#9467bd",
        },
    }

    worker_ids_to_plot = [0, 1, 2]

    save_dir = os.path.join(
        "eval_results",
        f"fatigue_case_seed_{case_config['seed']}_"
        f"{case_config['num_jobs']}x{case_config['num_machines']}x{case_config['num_workers']}"
    )
    os.makedirs(save_dir, exist_ok=True)

    method_results: Dict[str, Dict[str, Any]] = {}
    method_display_names = {}
    method_colors = {}

    print("Running fatigue case study...\n")

    for method_name, method_cfg in methods.items():
        print(f"Method: {method_name}")

        env = make_env(
            seed=case_config["seed"],
            num_jobs=case_config["num_jobs"],
            num_machines=case_config["num_machines"],
            num_workers=case_config["num_workers"],
            min_ops_per_job=case_config["min_ops_per_job"],
            max_ops_per_job=case_config["max_ops_per_job"],
            use_phase1=case_config["use_phase1"],
            use_phase2=case_config["use_phase2"],
        )

        if method_cfg["type"] == "rl":
            ckpt_path = method_cfg["actor_ckpt"]
            if not os.path.exists(ckpt_path):
                print(f"  [skip] checkpoint not found: {ckpt_path}")
                continue

            agent = build_rl_agent(
                actor_ckpt_path=ckpt_path,
                hidden_dim=case_config["hidden_dim"],
                num_layers=case_config["num_layers"],
            )
            result = run_case_rl(env, agent)

        elif method_cfg["type"] == "pdr":
            result = run_case_pdr(env, method_cfg["rule"])

        else:
            raise ValueError(f"Unknown method type: {method_cfg['type']}")

        method_results[method_name] = result
        method_display_names[method_name] = method_cfg["display_name"]
        method_colors[method_name] = method_cfg["color"]

        print(f"  makespan = {result['makespan']:.2f}")

    if not method_results:
        raise RuntimeError("No valid methods were executed.")

    global_max_makespan = max(result["makespan"] for result in method_results.values())

    for worker_id in worker_ids_to_plot:
        save_path = os.path.join(save_dir, f"worker_{worker_id}_fatigue_curves.png")
        plot_worker_fatigue_curves(
            save_path=save_path,
            worker_id=worker_id,
            method_results=method_results,
            method_display_names=method_display_names,
            method_colors=method_colors,
            global_max_makespan=global_max_makespan,
        )

    output = {
        "case_config": case_config,
        "global_max_makespan": round(float(global_max_makespan), 2),
        "worker_ids_to_plot": worker_ids_to_plot,
        "methods": {
            method_name: {
                "display_name": method_display_names[method_name],
                "color": method_colors[method_name],
                "makespan": round(float(result["makespan"]), 2),
                "trace": result["trace"],
                "final_schedule": result["final_schedule"],
            }
            for method_name, result in method_results.items()
        },
    }

    save_json = os.path.join(save_dir, "fatigue_case_study.json")
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved figures and case-study data to: {save_dir}")


if __name__ == "__main__":
    main()