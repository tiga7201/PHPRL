import os
import json
from typing import Dict, Any

import torch
import matplotlib.pyplot as plt

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state

from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from rl.sac_agent import SACAgent
from rl.pdr_baselines import select_pdr_action


# =========================
# matplotlib global config
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# basic paths
# =========================
def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_pgnn_phase1_path() -> str:
    return os.path.join(get_project_root(), "checkpoints", "pgnn_phase1.pt")


def get_pgnn_phase2_path() -> str:
    return os.path.join(get_project_root(), "checkpoints", "pgnn_phase2.pt")


# =========================
# environment construction
# =========================
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
        phase1_path = get_pgnn_phase1_path()
        if os.path.exists(phase1_path):
            env.load_pgnn_phase1(phase1_path, device="cpu")
        else:
            raise FileNotFoundError(f"Phase-1 PGNN checkpoint not found: {phase1_path}")

    if use_phase2:
        phase2_path = get_pgnn_phase2_path()
        if os.path.exists(phase2_path):
            env.load_pgnn_phase2(phase2_path, device="cpu")
        else:
            raise FileNotFoundError(f"Phase-2 PGNN checkpoint not found: {phase2_path}")

    return env


# =========================
# RL model loading
# =========================
def load_actor_checkpoint(actor, checkpoint_path: str, map_location: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    if isinstance(ckpt, dict):
        if "actor_state_dict" in ckpt:
            actor.load_state_dict(ckpt["actor_state_dict"])
        elif "model_state_dict" in ckpt:
            actor.load_state_dict(ckpt["model_state_dict"])
        else:
            # 如果字典就是纯 state_dict
            actor.load_state_dict(ckpt)
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


# =========================
# run one method on one instance
# =========================
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
        "worker_traces": {
            int(w): [(float(t), float(f)) for t, f in trace]
            for w, trace in env.worker_fatigue_traces.items()
        },
        "final_schedule": [str(x) for x in env.schedule],
        "final_worker_fatigue": {int(k): float(v) for k, v in env.worker_fatigue.items()},
        "final_worker_workload": {int(k): float(v) for k, v in env.worker_workload.items()},
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
        "worker_traces": {
            int(w): [(float(t), float(f)) for t, f in trace]
            for w, trace in env.worker_fatigue_traces.items()
        },
        "final_schedule": [str(x) for x in env.schedule],
        "final_worker_fatigue": {int(k): float(v) for k, v in env.worker_fatigue.items()},
        "final_worker_workload": {int(k): float(v) for k, v in env.worker_workload.items()},
    }


# =========================
# plot one figure per method
# =========================
def plot_method_fatigue_curves(
    save_path: str,
    result: Dict[str, Any],
    display_name: str,
    global_max_makespan: float,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(14, 9))

    worker_traces = result["worker_traces"]
    makespan = result["makespan"]

    # 这里不手动指定颜色，让 matplotlib 自动分配
    for worker_id, trace in worker_traces.items():
        times = [float(t) for t, _ in trace]
        fatigue = [float(f) for _, f in trace]

        plt.plot(
            times,
            fatigue,
            linewidth=2.0,
            alpha=0.95,
            label=f"$W_{worker_id+1}$",
        )

    plt.xlim(0.0, global_max_makespan+10)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Time", fontsize=40, labelpad=12)
    plt.ylabel("Fatigue level", fontsize=40, labelpad=12)
    plt.title(f"{display_name} | Makespan = {makespan:.2f}", fontsize=40, pad=12)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    plt.legend(handlelength=0.8, fontsize=40, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# main
# =========================
def main():
    # ===== 你主要改这里 =====
    case_config = {
        "seed": 550,
        "num_jobs": 5,
        "num_machines": 5,
        "num_workers": 3,   # 你可以改成 3、4、5、6...
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
            "display_name": "THGRL",
        },

        "full_sac": {
            "type": "rl",
            "actor_ckpt": "best_fixedscale_full_sac_actor.pt",
            "display_name": "HPRL",
        },
        "hagrl": {
            "type": "rl",
            "actor_ckpt": "checkpoints/archive/ckpt_step3_iter_0040.pt",
            "display_name": "HAGRL",
        },

        "thgrl": {
            "type": "rl",
            "actor_ckpt": "checkpoints/archive/ckpt_step3_iter_0250.pt",
            "display_name": "THGRL",
        },

        "spt": {
            "type": "pdr",
            "rule": "SPT",
            "display_name": "SPT",
        },
        "fifo": {
            "type": "pdr",
            "rule": "FIFO",
            "display_name": "FIFO",
        },
        "mwkr": {
            "type": "pdr",
            "rule": "MWKR",
            "display_name": "MWKR",
        },
    }

    # 文件名映射：方便你后续排版
    filename_map = {
        "pref_3stage_parallel_best": "proposed.png",
        "full_sac": "full_sac.png",
        "spt": "spt.png",
        "fifo": "fifo.png",
        "mwkr": "mwkr.png",
    }

    save_dir = os.path.join(
        "eval_results",
        f"fatigue_case_seed_{case_config['seed']}_"
        f"{case_config['num_jobs']}x{case_config['num_machines']}x{case_config['num_workers']}"
    )
    os.makedirs(save_dir, exist_ok=True)

    method_results: Dict[str, Dict[str, Any]] = {}
    method_display_names: Dict[str, str] = {}

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

        print(f"  makespan = {result['makespan']:.2f}")
        print(f"  final fatigue = {result['final_worker_fatigue']}")
        print(f"  final workload = {result['final_worker_workload']}")

    if not method_results:
        raise RuntimeError("No valid methods were executed.")

    # 统一横轴到该实例下所有方法的最大 makespan
    global_max_makespan = max(result["makespan"] for result in method_results.values())

    # 每个方法单独输出一张图
    for method_name, result in method_results.items():
        save_path = os.path.join(
            save_dir,
            filename_map.get(method_name, f"{method_name}.png")
        )

        plot_method_fatigue_curves(
            save_path=save_path,
            result=result,
            display_name=method_display_names[method_name],
            global_max_makespan=global_max_makespan,
        )

    # 保存 summary
    output = {
        "case_config": case_config,
        "global_max_makespan": round(float(global_max_makespan), 2),
        "methods": {
            method_name: {
                "display_name": method_display_names[method_name],
                "makespan": round(float(result["makespan"]), 2),
                "worker_traces": result["worker_traces"],
                "final_worker_fatigue": result["final_worker_fatigue"],
                "final_worker_workload": result["final_worker_workload"],
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