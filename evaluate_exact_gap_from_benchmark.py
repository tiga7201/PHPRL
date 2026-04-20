import torch

from env.instance_generator import load_instance_dataset
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from exact.brute_force_solver import BruteForceSolver

from models.actor_shyper import SHyperActor
from models.actor_shyper_gated import SHyperActorGated
from models.actor_shyper_full import SHyperActorFull


def build_actor(model_type: str, actor_path: str, hidden_dim=64, num_layers=2):
    if model_type == "simplified":
        actor = SHyperActor(hidden_dim=hidden_dim)
    elif model_type == "gated":
        actor = SHyperActorGated(hidden_dim=hidden_dim)
    elif model_type == "full":
        actor = SHyperActorFull(hidden_dim=hidden_dim, num_layers=num_layers)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    actor.eval()
    return actor


def solve_with_actor_greedy(env, actor):
    state = env.reset()
    done = False

    while not done:
        graph_state = build_hypergraph_state(env)
        decision = actor.select_greedy_action(graph_state)
        action = decision["action"]
        state, reward, done, info = env.step(action)

    return info["makespan"], env.schedule


def solve_exact(env):
    solver = BruteForceSolver(env)
    result = solver.solve()
    return (
        result["best_makespan"],
        result["best_schedule"],
        result["best_action_seq"],
        result["num_nodes_searched"],
    )


def evaluate_from_benchmark(
    benchmark_path: str,
    model_type: str,
    actor_path: str,
    hidden_dim=64,
    num_layers=2,
):
    actor = build_actor(
        model_type=model_type,
        actor_path=actor_path,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    instances = load_instance_dataset(benchmark_path)
    results = []

    for idx, instance in enumerate(instances):
        env_model = FJSPWFEnv(instance)
        model_makespan, model_schedule = solve_with_actor_greedy(env_model, actor)

        env_exact = FJSPWFEnv(instance)
        optimal_makespan, optimal_schedule, optimal_actions, searched_nodes = solve_exact(env_exact)

        gap = (model_makespan - optimal_makespan) / optimal_makespan

        results.append({
            "instance_id": idx,
            "model_makespan": model_makespan,
            "optimal_makespan": optimal_makespan,
            "gap": gap,
            "searched_nodes": searched_nodes,
            "model_schedule": model_schedule,
            "optimal_schedule": optimal_schedule,
            "optimal_actions": optimal_actions,
        })

    return results


def print_summary(results, title=""):
    if title:
        print(title)
        print()

    avg_gap = 0.0
    avg_model = 0.0
    avg_opt = 0.0

    for item in results:
        avg_gap += item["gap"]
        avg_model += item["model_makespan"]
        avg_opt += item["optimal_makespan"]

        print(
            f"Instance {item['instance_id']} | "
            f"model={item['model_makespan']:.4f} | "
            f"optimal={item['optimal_makespan']:.4f} | "
            f"gap={item['gap'] * 100:.2f}% | "
            f"searched_nodes={item['searched_nodes']}"
        )

    n = len(results)
    avg_gap /= n
    avg_model /= n
    avg_opt /= n

    print("\nSummary:")
    print(f"Average model makespan:   {avg_model:.4f}")
    print(f"Average optimal makespan: {avg_opt:.4f}")
    print(f"Average gap:              {avg_gap * 100:.2f}%")

    worst = max(results, key=lambda x: x["gap"])
    print("\nWorst-case instance:")
    print(
        f"Instance {worst['instance_id']} | "
        f"model={worst['model_makespan']:.4f} | "
        f"optimal={worst['optimal_makespan']:.4f} | "
        f"gap={worst['gap'] * 100:.2f}%"
    )


def main():
    benchmark_path = "benchmark_exact_3x3x3.json"

    # 改这里来评估不同模型
    model_type = "full"
    actor_path = "best_fixedscale_full_sac_actor.pt"

    results = evaluate_from_benchmark(
        benchmark_path=benchmark_path,
        model_type=model_type,
        actor_path=actor_path,
        hidden_dim=64,
        num_layers=2,
    )

    print_summary(results, title=f"Exact-gap evaluation ({model_type})")


if __name__ == "__main__":
    main()