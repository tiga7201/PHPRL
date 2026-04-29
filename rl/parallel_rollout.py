from concurrent.futures import ProcessPoolExecutor, as_completed

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
import os

def _collect_trajectories_for_one_instance_worker(
    seed,
    actor_state_dict,
    actor_class,
    actor_kwargs,
    num_trajectories,
    num_jobs,
    num_machines,
    num_workers,
    min_ops_per_job,
    max_ops_per_job,
):
    """
    Worker function:
      - rebuild actor locally
      - rebuild env locally
      - sample multiple trajectories for one instance
      - return raw trajectory dicts
    """
    actor = actor_class(**actor_kwargs)
    actor.load_state_dict(actor_state_dict)
    actor.eval()

    instance = generate_random_instance(
        seed=seed,
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_workers=num_workers,
        min_ops_per_job=min_ops_per_job,
        max_ops_per_job=max_ops_per_job,
    )
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pgnn_path = os.path.join(project_root, "checkpoints", "pgnn_phase1.pt")
    env = FJSPWFEnv(instance)
    env.load_pgnn_phase1(pgnn_path, device="cpu")

    results = []

    for traj_idx in range(num_trajectories):
        env.reset()
        done = False

        graph_states = []
        edge_indices = []
        actions = []
        rewards = []
        final_makespan = None

        while not done:
            graph_state = build_hypergraph_state(env)
            decision = actor.sample_action(graph_state)

            action = decision["action"]
            edge_idx = decision["edge_idx"]

            _, reward, done, info = env.step(action)

            graph_states.append(graph_state)
            edge_indices.append(edge_idx)
            actions.append(action)
            rewards.append(float(reward))
            final_makespan = float(info["makespan"])

        results.append({
            "instance_id": f"inst_{seed}",
            "traj_id": f"inst_{seed}_traj_{traj_idx}",
            "graph_states": graph_states,
            "edge_indices": edge_indices,
            "actions": actions,
            "rewards": rewards,
            "makespan": final_makespan,
        })

    return results


def collect_trajectories_parallel(
    seeds,
    actor,
    actor_class,
    actor_kwargs,
    num_trajectories_per_instance,
    num_jobs=3,
    num_machines=3,
    num_workers=3,
    min_ops_per_job=2,
    max_ops_per_job=4,
    max_workers=4,
):
    """
    Parallel rollout for a batch of fixed instances.

    Returns:
      all_traj_dicts: list of trajectory dicts
    """
    actor_state_dict = actor.state_dict()
    all_traj_dicts = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for seed in seeds:
            fut = executor.submit(
                _collect_trajectories_for_one_instance_worker,
                seed,
                actor_state_dict,
                actor_class,
                actor_kwargs,
                num_trajectories_per_instance,
                num_jobs,
                num_machines,
                num_workers,
                min_ops_per_job,
                max_ops_per_job,
            )
            futures.append(fut)

        for fut in as_completed(futures):
            all_traj_dicts.extend(fut.result())

    return all_traj_dicts