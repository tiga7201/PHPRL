from copy import deepcopy

from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from exact.brute_force_solver import BruteForceSolver


def build_optimal_demo_dataset():
    """
    Build supervised samples from the exact optimal action sequence.

    Returns:
        dataset: list of dicts
            each dict contains:
            - graph_state
            - optimal_action
            - optimal_edge_idx
    """
    instance = create_demo_instance()
    env = FJSPWFEnv(instance)

    solver = BruteForceSolver(env)
    result = solver.solve()

    optimal_actions = result["best_action_seq"]

    rollout_env = FJSPWFEnv(create_demo_instance())
    rollout_env.reset()

    dataset = []

    for action in optimal_actions:
        graph_state = build_hypergraph_state(rollout_env)
        edge_idx = graph_state["action_to_edge"][action]

        dataset.append({
            "graph_state": graph_state,
            "optimal_action": action,
            "optimal_edge_idx": edge_idx,
        })

        rollout_env.step(action)

    return dataset, result


def main():
    dataset, result = build_optimal_demo_dataset()

    print("Optimal makespan:", result["best_makespan"])
    print("Num supervised samples:", len(dataset))

    for i, item in enumerate(dataset):
        print(f"\nSample {i}")
        print("optimal_action:", item["optimal_action"])
        print("optimal_edge_idx:", item["optimal_edge_idx"])
        print("valid_action_mask:", item["graph_state"]["valid_action_mask"])


if __name__ == "__main__":
    main()