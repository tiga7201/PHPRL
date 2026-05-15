import os
import json
from typing import Dict, Any, List
import torch

import matplotlib.pyplot as plt

from env.instance_generator import generate_random_instance, instance_to_dict
from env.fjspwf_env import FJSPWFEnv
from rl.pdr_baselines import select_pdr_action

from utils.graph_builder import build_hypergraph_state
from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from rl.sac_agent import SACAgent

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


def build_rl_agent(
    actor_ckpt_path: str,
    hidden_dim: int = 64,
    num_layers: int = 2,
):
    actor = SHyperActorFull(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    ckpt = torch.load(actor_ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "actor_state_dict" in ckpt:
            actor.load_state_dict(ckpt["actor_state_dict"])
        else:
            actor.load_state_dict(ckpt)
    else:
        actor.load_state_dict(ckpt)

    actor.eval()

    q1 = SHyperQCriticFull(hidden_dim=hidden_dim, num_layers=num_layers)
    q2 = SHyperQCriticFull(hidden_dim=hidden_dim, num_layers=num_layers)

    agent = SACAgent(
        actor=actor,
        q1=q1,
        q2=q2,
        device="cpu",
    )

    return agent


def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_pgnn_phase1_path() -> str:
    return os.path.join(get_project_root(), "checkpoints", "pgnn_phase1.pt")


def get_pgnn_phase2_path() -> str:
    return os.path.join(get_project_root(), "checkpoints", "pgnn_phase2.pt")


def build_fixed_case_instance(worker_cases):
    instance = generate_random_instance(
        seed=2026,
        num_jobs=11,
        num_machines=6,
        num_workers=6,
        min_ops_per_job=3,
        max_ops_per_job=7,
        proc_time_low=5,
        proc_time_high=30,
    )

    instance = repair_worker_compatibility_for_nested_cases(
        instance=instance,
        worker_cases=worker_cases,
        seed=2026,
    )

    instance = repair_standard_processing_time_by_operation(instance)

    return instance

def get_min_worker_case(worker_cases):
    """
    For nested worker cases, return the smallest available-worker set.
    Example:
        W3=[0,1,2], W4=[0,1,2,3] -> return [0,1,2]
    """
    return min(worker_cases.values(), key=len)


def repair_worker_compatibility_for_nested_cases(instance, worker_cases, seed=2026):
    """
    For nested worker cases, only repair compatibility against the smallest case.
    """
    import random
    rng = random.Random(seed)

    core_workers = list(get_min_worker_case(worker_cases))

    for job in instance.jobs:
        for op in job:
            compatible_set = set(op.compatible_workers)

            if compatible_set.isdisjoint(core_workers):
                selected_worker = rng.choice(core_workers)
                compatible_set.add(selected_worker)

            op.compatible_workers = sorted(list(compatible_set))

    return instance

def repair_standard_processing_time_by_operation(instance):
    """
    For aircraft assembly case:
    each operation has one standard processing time, independent of machine-worker pair.
    """
    for job in instance.jobs:
        for op in job:
            all_times = list(op.base_processing_times.values())

            if not all_times:
                raise ValueError(f"Operation {op.op_id} has no base processing time.")

            standard_time = round(sum(all_times) / len(all_times))

            for key in list(op.base_processing_times.keys()):
                op.base_processing_times[key] = standard_time

    return instance


def make_case_env(instance, available_workers: List[int], use_fatigue: bool):
    env = FJSPWFEnv(
        instance=instance,
        available_workers=available_workers,
        use_fatigue=use_fatigue,
    )

    if use_fatigue:
        env.load_pgnn_phase1(get_pgnn_phase1_path(), device="cpu")

        phase2_path = get_pgnn_phase2_path()
        if os.path.exists(phase2_path):
            env.load_pgnn_phase2(phase2_path, device="cpu")

    return env


def run_pdr_case(env: FJSPWFEnv, rule: str = "SPT") -> Dict[str, Any]:
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
        "final_worker_fatigue": {int(k): float(v) for k, v in env.worker_fatigue.items()},
        "final_worker_workload": {int(k): float(v) for k, v in env.worker_workload.items()},
        "final_schedule": [str(x) for x in env.schedule],
    }

def run_proposed_case(env, agent):
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
        "final_worker_fatigue": {
            int(k): float(v)
            for k, v in env.worker_fatigue.items()
        },
        "final_worker_workload": {
            int(k): float(v)
            for k, v in env.worker_workload.items()
        },
    }

def plot_case_worker_fatigue(
    save_path: str,
    result: Dict[str, Any],
    available_workers: List[int],
    title: str,
    global_xmax: float,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(14, 9))

    for worker_id in available_workers:
        trace = result["worker_traces"][worker_id]
        times = [float(t) for t, _ in trace]
        fatigue = [float(f) for _, f in trace]

        plt.plot(
            times,
            fatigue,
            linewidth=2.0,
            alpha=0.95,
            label=f"$W_{worker_id+1}$",
        )

    plt.xlim(0.0, global_xmax+5)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Time (min)", fontsize=40, labelpad=12)
    plt.ylabel("Fatigue level", fontsize=40, labelpad=12)
    plt.title(title, fontsize=40, pad=12)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    plt.legend(handlelength=0.8, fontsize=40, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_makespan_bar(save_path: str, summary_rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = [
        f"{row['case_name']}\n{row['mode']}"
        for row in summary_rows
    ]
    values = [row["makespan"] for row in summary_rows]

    plt.figure(figsize=(10, 5.5))
    plt.bar(labels, values)
    plt.ylabel("Makespan", fontsize=13, labelpad=10)
    plt.xticks(fontsize=10, rotation=30, ha="right")
    plt.yticks(fontsize=11)
    plt.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    rule = "SPT"
    # actor_ckpt_path = "best_pref3stage_parallel_actor.pt"

    worker_cases = {
        # "W2": [0, 1],
        "Three worker groups": [0, 1, 2],
        "Four worker groups": [0, 1, 2, 3],
        "Five worker groups": [0, 1, 2, 3, 4],
        "Six worker groups": [0, 1, 2, 3, 4, 5],
    }
    instance = build_fixed_case_instance(worker_cases)

    num_ops_per_job = [len(job) for job in instance.jobs]
    total_ops = sum(num_ops_per_job)

    print("\n=== Fixed case instance information ===")
    print(f"Number of jobs: {instance.num_jobs}")
    print(f"Number of machines: {instance.num_machines}")
    print(f"Number of workers: {instance.num_workers}")
    print(f"Operations per job: {num_ops_per_job}")
    print(f"Total number of operations: {total_ops}")
    print("======================================\n")

    fatigue_modes = {
        "standard_time": False,
        "fatigue_aware": True,
    }

    save_dir = os.path.join("eval_results", "aircraft_case_worker_availability")
    os.makedirs(save_dir, exist_ok=True)

    all_results = {}
    summary_rows = []

    for case_name, available_workers in worker_cases.items():
        for mode_name, use_fatigue in fatigue_modes.items():
            print(f"Running case={case_name}, mode={mode_name}, workers={available_workers}")

            try:
                env = make_case_env(
                    instance=instance,
                    available_workers=available_workers,
                    use_fatigue=use_fatigue,
                )
            except ValueError as e:
                print(f"  [skip infeasible] {e}")
                continue

            agent = build_rl_agent(
                actor_ckpt_path="best_pref3stage_parallel_actor.pt",
                hidden_dim=64,
                num_layers=2,
            )
            # result = run_proposed_case(env, agent)
            result = run_pdr_case(env, rule=rule)

            result_key = f"{case_name}_{mode_name}"

            all_results[result_key] = {
                "case_name": case_name,
                "available_workers": available_workers,
                "mode": mode_name,
                "use_fatigue": use_fatigue,
                "method": "Proposed",
                # "actor_ckpt": actor_ckpt_path,
                **result,
            }

            summary_rows.append({
                "case_name": case_name,
                "available_workers": available_workers,
                "mode": mode_name,
                "use_fatigue": use_fatigue,
                "makespan": round(float(result["makespan"]), 2),
                "final_worker_fatigue": result["final_worker_fatigue"],
                "final_worker_workload": result["final_worker_workload"],
            })

            print(f"  makespan={result['makespan']:.2f}")

    if not all_results:
        raise RuntimeError("No feasible case was executed.")

    global_xmax = max(v["makespan"] for v in all_results.values())

    for result_key, result in all_results.items():
        if not result["use_fatigue"]:
            continue

        available_workers = result["available_workers"]

        title = (
            f"{result['case_name']} | "
            f"Makespan = {result['makespan']:.2f}"
        )

        save_path = os.path.join(save_dir, f"{result_key}_fatigue_curves.png")

        plot_case_worker_fatigue(
            save_path=save_path,
            result=result,
            available_workers=available_workers,
            title=title,
            global_xmax=global_xmax,
        )

    plot_makespan_bar(
        save_path=os.path.join(save_dir, "makespan_comparison.png"),
        summary_rows=summary_rows,
    )

    output = {
        "case_description": {
            "instance_type": "fixed-seed random aircraft assembly case",
            "method": "Proposed",
            # "actor_ckpt": actor_ckpt_path,
            "worker_cases": worker_cases,
            "fatigue_modes": fatigue_modes,
        },
        "instance_statistics": {
            "num_jobs": instance.num_jobs,
            "num_machines": instance.num_machines,
            "num_workers": instance.num_workers,
            "operations_per_job": num_ops_per_job,
            "total_operations": total_ops,
        },
        "instance": instance_to_dict(instance),
        "summary": summary_rows,
        "results": all_results,
    }

    save_json = os.path.join(save_dir, "aircraft_case_worker_availability_results.json")
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to: {save_dir}")


if __name__ == "__main__":
    main()