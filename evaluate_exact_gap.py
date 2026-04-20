import torch

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from exact.brute_force_solver import BruteForceSolver
from models.actor_shyper import SHyperActor


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


def evaluate_actor_against_exact(
    actor_path,
    hidden_dim=64,
    eval_seeds=None,
):
    if eval_seeds is None:
        eval_seeds = [100, 101, 102, 103, 104]

    actor = SHyperActor(hidden_dim=hidden_dim)
    actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    actor.eval()

    results = []

    for seed in eval_seeds:
        instance = generate_random_instance(seed=seed)

        env_model = FJSPWFEnv(instance)
        model_makespan, model_schedule = solve_with_actor_greedy(env_model, actor)

        env_exact = FJSPWFEnv(instance)
        optimal_makespan, optimal_schedule, optimal_actions, searched_nodes = solve_exact(env_exact)

        gap = (model_makespan - optimal_makespan) / optimal_makespan

        results.append({
            "seed": seed,
            "model_makespan": model_makespan,
            "optimal_makespan": optimal_makespan,
            "gap": gap,
            "searched_nodes": searched_nodes,
            "model_schedule": model_schedule,
            "optimal_schedule": optimal_schedule,
            "optimal_actions": optimal_actions,
        })

    return results


def main():
    actor_path = "best_fixedscale_sac_shyper_actor.pt"
    eval_seeds = [100, 101, 102, 103, 104]

    results = evaluate_actor_against_exact(
        actor_path=actor_path,
        hidden_dim=64,
        eval_seeds=eval_seeds,
    )

    print("Exact-gap evaluation results:\n")

    avg_gap = 0.0
    avg_model = 0.0
    avg_opt = 0.0

    for item in results:
        avg_gap += item["gap"]
        avg_model += item["model_makespan"]
        avg_opt += item["optimal_makespan"]

        print(
            f"Seed {item['seed']} | "
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

    print("\nWorst-case instance:")
    worst = max(results, key=lambda x: x["gap"])
    print(
        f"Seed {worst['seed']} | "
        f"model={worst['model_makespan']:.4f} | "
        f"optimal={worst['optimal_makespan']:.4f} | "
        f"gap={worst['gap'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()