import os
import json
import math
from typing import Dict, List, Any

import torch

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state

from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from rl.sac_agent import SACAgent
from rl.pdr_baselines import select_pdr_action


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
        else:
            print(f"[warn] phase2 checkpoint not found: {phase2_path}. Fallback to phase1 only.")

    return env


def load_actor_checkpoint(actor, checkpoint_path, map_location="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
        actor.load_state_dict(ckpt["actor_state_dict"])
    else:
        actor.load_state_dict(ckpt)


def build_rl_agent(
    actor_ckpt_path: str,
    hidden_dim: int = 64,
    num_layers: int = 2,
):
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


def snapshot_env(env: FJSPWFEnv, step_idx: int) -> Dict[str, Any]:
    fatigue_dict = {int(k): float(v) for k, v in env.worker_fatigue.items()}
    workload_dict = {int(k): float(v) for k, v in env.worker_workload.items()}

    fatigue_values = list(fatigue_dict.values())
    workload_values = list(workload_dict.values())

    avg_fatigue = sum(fatigue_values) / len(fatigue_values) if fatigue_values else 0.0
    max_fatigue = max(fatigue_values) if fatigue_values else 0.0
    min_fatigue = min(fatigue_values) if fatigue_values else 0.0
    fatigue_range = max_fatigue - min_fatigue
    fatigue_std = (
        math.sqrt(sum((x - avg_fatigue) ** 2 for x in fatigue_values) / len(fatigue_values))
        if fatigue_values else 0.0
    )

    avg_workload = sum(workload_values) / len(workload_values) if workload_values else 0.0
    max_workload = max(workload_values) if workload_values else 0.0

    return {
        "step_idx": int(step_idx),
        "current_time": float(env.current_time),
        "worker_fatigue": fatigue_dict,
        "worker_workload": workload_dict,
        "avg_fatigue": float(avg_fatigue),
        "max_fatigue": float(max_fatigue),
        "min_fatigue": float(min_fatigue),
        "fatigue_range": float(fatigue_range),
        "fatigue_std": float(fatigue_std),
        "avg_workload": float(avg_workload),
        "max_workload": float(max_workload),
    }


def run_episode_rl_with_trace(env: FJSPWFEnv, agent) -> Dict[str, Any]:
    env.reset()
    done = False
    step_idx = 0
    trace = [snapshot_env(env, step_idx=0)]
    actions = []
    last_info = None

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            old_time = env.current_time
            env._advance_to_next_event()
            step_idx += 1
            trace.append(snapshot_env(env, step_idx=step_idx))
            actions.append({
                "type": "advance",
                "time_from": float(old_time),
                "time_to": float(env.current_time),
            })
            continue

        graph_state = build_hypergraph_state(env)
        decision = agent.actor.select_greedy_action(graph_state)
        action = decision["action"]

        _, _, done, info = env.step(action)
        last_info = info
        step_idx += 1

        actions.append({
            "type": "dispatch",
            "action": tuple(int(x) for x in action),
            "start_time": float(info["action_start_time"]),
            "end_time": float(info["action_end_time"]),
            "proc_time": float(info["proc_time"]),
            "fatigue_before": float(info["fatigue_before"]),
            "fatigue_after": float(info["fatigue_after"]),
        })
        trace.append(snapshot_env(env, step_idx=step_idx))

    return {
        "makespan": float(last_info["makespan"]),
        "trace": trace,
        "actions": actions,
        "final_schedule": [str(x) for x in env.schedule],
    }


def run_episode_pdr_with_trace(env: FJSPWFEnv, rule: str) -> Dict[str, Any]:
    env.reset()
    done = False
    step_idx = 0
    trace = [snapshot_env(env, step_idx=0)]
    actions = []
    last_info = None

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            old_time = env.current_time
            env._advance_to_next_event()
            step_idx += 1
            trace.append(snapshot_env(env, step_idx=step_idx))
            actions.append({
                "type": "advance",
                "time_from": float(old_time),
                "time_to": float(env.current_time),
            })
            continue

        action = select_pdr_action(env, rule)
        _, _, done, info = env.step(action)
        last_info = info
        step_idx += 1

        actions.append({
            "type": "dispatch",
            "rule": rule,
            "action": tuple(int(x) for x in action),
            "start_time": float(info["action_start_time"]),
            "end_time": float(info["action_end_time"]),
            "proc_time": float(info["proc_time"]),
            "fatigue_before": float(info["fatigue_before"]),
            "fatigue_after": float(info["fatigue_after"]),
        })
        trace.append(snapshot_env(env, step_idx=step_idx))

    return {
        "makespan": float(last_info["makespan"]),
        "trace": trace,
        "actions": actions,
        "final_schedule": [str(x) for x in env.schedule],
    }


def summarize_trace(run_result: Dict[str, Any]) -> Dict[str, Any]:
    trace = run_result["trace"]
    final_state = trace[-1]

    avg_fatigue_over_time = sum(x["avg_fatigue"] for x in trace) / len(trace)
    max_fatigue_over_time = max(x["max_fatigue"] for x in trace)
    avg_fatigue_std_over_time = sum(x["fatigue_std"] for x in trace) / len(trace)

    return {
        "makespan": float(run_result["makespan"]),
        "final_avg_fatigue": float(final_state["avg_fatigue"]),
        "final_max_fatigue": float(final_state["max_fatigue"]),
        "final_fatigue_std": float(final_state["fatigue_std"]),
        "final_fatigue_range": float(final_state["fatigue_range"]),
        "avg_fatigue_over_time": float(avg_fatigue_over_time),
        "max_fatigue_over_time": float(max_fatigue_over_time),
        "avg_fatigue_std_over_time": float(avg_fatigue_std_over_time),
    }


def aggregate_seed_results(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def avg(key: str) -> float:
        return float(sum(x["summary"][key] for x in seed_results) / len(seed_results))

    return {
        "avg_makespan": avg("makespan"),
        "avg_final_avg_fatigue": avg("final_avg_fatigue"),
        "avg_final_max_fatigue": avg("final_max_fatigue"),
        "avg_final_fatigue_std": avg("final_fatigue_std"),
        "avg_final_fatigue_range": avg("final_fatigue_range"),
        "avg_avg_fatigue_over_time": avg("avg_fatigue_over_time"),
        "avg_max_fatigue_over_time": avg("max_fatigue_over_time"),
        "avg_avg_fatigue_std_over_time": avg("avg_fatigue_std_over_time"),
    }


def main():
    eval_config = {
        "seeds": [100, 101, 102, 103, 104],
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
        "fifo": {
            "type": "pdr",
            "rule": "FIFO",
        },
        "spt": {
            "type": "pdr",
            "rule": "SPT",
        },
        "mwkr": {
            "type": "pdr",
            "rule": "MWKR",
        },
        "pref_3stage_parallel_best": {
            "type": "rl",
            "actor_ckpt": "best_pref3stage_parallel_actor.pt",
        },
    }

    all_results = {
        "eval_config": eval_config,
        "methods": {},
    }

    for method_name, method_cfg in methods.items():
        print(f"\nAnalyzing method: {method_name}")
        seed_results = []

        agent = None
        if method_cfg["type"] == "rl":
            ckpt_path = method_cfg["actor_ckpt"]
            if not os.path.exists(ckpt_path):
                print(f"[skip] checkpoint not found: {ckpt_path}")
                continue
            agent = build_rl_agent(
                actor_ckpt_path=ckpt_path,
                hidden_dim=eval_config["hidden_dim"],
                num_layers=eval_config["num_layers"],
            )

        for seed in eval_config["seeds"]:
            env = make_env(
                seed=seed,
                num_jobs=eval_config["num_jobs"],
                num_machines=eval_config["num_machines"],
                num_workers=eval_config["num_workers"],
                min_ops_per_job=eval_config["min_ops_per_job"],
                max_ops_per_job=eval_config["max_ops_per_job"],
                use_phase1=eval_config["use_phase1"],
                use_phase2=eval_config["use_phase2"],
            )

            if method_cfg["type"] == "pdr":
                run_result = run_episode_pdr_with_trace(env, method_cfg["rule"])
            elif method_cfg["type"] == "rl":
                run_result = run_episode_rl_with_trace(env, agent)
            else:
                raise ValueError(f"Unknown method type: {method_cfg['type']}")

            summary = summarize_trace(run_result)

            seed_result = {
                "seed": int(seed),
                "summary": summary,
                "trace": run_result["trace"],
                "actions": run_result["actions"],
                "final_schedule": run_result["final_schedule"],
            }
            seed_results.append(seed_result)

            print(
                f"  seed={seed} | "
                f"makespan={summary['makespan']:.4f} | "
                f"final_avg_fatigue={summary['final_avg_fatigue']:.4f} | "
                f"final_max_fatigue={summary['final_max_fatigue']:.4f} | "
                f"final_fatigue_std={summary['final_fatigue_std']:.4f}"
            )

        aggregate = aggregate_seed_results(seed_results)

        all_results["methods"][method_name] = {
            "type": method_cfg["type"],
            "seed_results": seed_results,
            "aggregate": aggregate,
        }

        print(
            f"Summary {method_name} | "
            f"avg_makespan={aggregate['avg_makespan']:.4f} | "
            f"avg_final_avg_fatigue={aggregate['avg_final_avg_fatigue']:.4f} | "
            f"avg_final_max_fatigue={aggregate['avg_final_max_fatigue']:.4f} | "
            f"avg_final_fatigue_std={aggregate['avg_final_fatigue_std']:.4f}"
        )

    os.makedirs("eval_results", exist_ok=True)
    save_path = os.path.join("eval_results", "analysis_methods_fatigue.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()