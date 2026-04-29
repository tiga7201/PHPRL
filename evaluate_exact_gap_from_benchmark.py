import json
import os
import torch

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state

from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from rl.sac_agent import SACAgent

def get_pgnn_checkpoint_path():
    project_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(project_root, "checkpoints", "pgnn_phase1.pt")

def make_env_from_seed(
    seed,
    num_jobs=3,
    num_machines=3,
    num_workers=3,
    min_ops_per_job=2,
    max_ops_per_job=4,
):
    instance = generate_random_instance(
        seed=seed,
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_workers=num_workers,
        min_ops_per_job=min_ops_per_job,
        max_ops_per_job=max_ops_per_job,
    )
    env = FJSPWFEnv(instance)
    env.load_pgnn_phase1(get_pgnn_checkpoint_path(), device="cpu")
    return env


def run_greedy_episode(env, agent):
    env.reset()
    done = False
    last_info = None

    while not done:
        graph_state = build_hypergraph_state(env)
        decision = agent.actor.select_greedy_action(graph_state)
        action = decision["action"]
        _, _, done, info = env.step(action)
        last_info = info

    return float(last_info["makespan"])


def load_actor_checkpoint(actor, checkpoint_path, map_location="cpu"):
    """
    Support two formats:
    1. pure actor state_dict
    2. full training checkpoint containing 'actor_state_dict'
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
        actor.load_state_dict(ckpt["actor_state_dict"])
    else:
        actor.load_state_dict(ckpt)


def build_agent(hidden_dim=64, num_layers=2):
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
    return agent


def main():
    # ===== config =====
    benchmark_path = "benchmark_exact_3x3x3.json"

    # 可改成：
    # actor_path = "best_pref3stage_parallel_actor.pt"
    # 或：
    # actor_path = "checkpoints/archive/ckpt_step3_iter_0040.pt"
    actor_path = "best_pref3stage_parallel_actor.pt"

    hidden_dim = 64
    num_layers = 2

    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Checkpoint file not found: {actor_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    agent = build_agent(hidden_dim=hidden_dim, num_layers=num_layers)
    load_actor_checkpoint(agent.actor, actor_path, map_location="cpu")
    agent.actor.eval()

    results = []

    for idx, item in enumerate(benchmark["instances"]):
        seed = int(item["seed"])
        optimal = float(item["optimal_makespan"])

        env = make_env_from_seed(
            seed=seed,
            num_jobs=int(item["num_jobs"]),
            num_machines=int(item["num_machines"]),
            num_workers=int(item["num_workers"]),
            min_ops_per_job=int(item["min_ops_per_job"]),
            max_ops_per_job=int(item["max_ops_per_job"]),
        )

        model_makespan = run_greedy_episode(env, agent)
        gap = (model_makespan - optimal) / optimal * 100.0

        results.append({
            "instance_index": idx,
            "seed": seed,
            "model_makespan": float(model_makespan),
            "optimal_makespan": float(optimal),
            "gap_percent": float(gap),
        })

        print(
            f"Instance {idx} | "
            f"seed={seed} | "
            f"model={model_makespan:.4f} | "
            f"optimal={optimal:.4f} | "
            f"gap={gap:.2f}%"
        )

    avg_model = sum(x["model_makespan"] for x in results) / len(results)
    avg_opt = sum(x["optimal_makespan"] for x in results) / len(results)
    avg_gap = sum(x["gap_percent"] for x in results) / len(results)
    worst_item = max(results, key=lambda x: x["gap_percent"])
    solved_optimally = sum(1 for x in results if abs(x["gap_percent"]) < 1e-9)
    solved_ratio = solved_optimally / len(results)

    print("\nSummary:")
    print(f"Average model makespan:   {avg_model:.4f}")
    print(f"Average optimal makespan: {avg_opt:.4f}")
    print(f"Average gap:              {avg_gap:.2f}%")
    print(
        f"Solved optimally ratio:   {solved_optimally}/{len(results)} "
        f"({solved_ratio * 100:.2f}%)"
    )
    print("Worst-case instance:")
    print(
        f"Instance {worst_item['instance_index']} | "
        f"seed={worst_item['seed']} | "
        f"model={worst_item['model_makespan']:.4f} | "
        f"optimal={worst_item['optimal_makespan']:.4f} | "
        f"gap={worst_item['gap_percent']:.2f}%"
    )

    os.makedirs("eval_results", exist_ok=True)
    output = {
        "benchmark_path": benchmark_path,
        "actor_path": actor_path,
        "summary": {
            "average_model_makespan": avg_model,
            "average_optimal_makespan": avg_opt,
            "average_gap_percent": avg_gap,
            "solved_optimally_count": solved_optimally,
            "solved_optimally_ratio": solved_ratio,
            "worst_case_instance": worst_item,
        },
        "results": results,
    }

    save_name = os.path.join("eval_results", "evaluation_exact_gap.json")
    with open(save_name, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {save_name}")


if __name__ == "__main__":
    main()