from typing import List, Dict, Any

from rl.trajectory_collector import collect_multiple_trajectories_for_env
from rl.preference_dataset import build_preference_pairs
from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv


def make_env(
    seed=None,
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
    return FJSPWFEnv(instance)


def build_training_batch_envs(
    start_seed,
    batch_instance_size,
    num_jobs=3,
    num_machines=3,
    num_workers=3,
    min_ops_per_job=2,
    max_ops_per_job=4,
):
    envs = []
    instance_ids = []

    for i in range(batch_instance_size):
        seed = start_seed + i
        env = make_env(
            seed=seed,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
        )
        envs.append(env)
        instance_ids.append(f"inst_seed_{seed}")

    return envs, instance_ids


def collect_preference_data_for_batch(
    envs: List[FJSPWFEnv],
    instance_ids: List[str],
    agent,
    num_trajectories_per_instance: int,
    min_gap: float = 0.0,
):
    """
    For a batch of fixed instances:
      - collect multiple trajectories for each instance
      - build preference pairs within each instance

    Returns:
      {
        "trajectory_groups": List[List[Trajectory]],
        "pair_groups": List[List[PreferencePair]],
        "all_pairs": List[PreferencePair],
        "summary": List[dict]
      }
    """
    trajectory_groups = []
    pair_groups = []
    all_pairs = []
    summary = []

    for env, instance_id in zip(envs, instance_ids):
        trajs = collect_multiple_trajectories_for_env(
            env=env,
            agent=agent,
            instance_id=instance_id,
            num_trajectories=num_trajectories_per_instance,
        )

        pairs = build_preference_pairs(trajs, min_gap=min_gap)

        trajectory_groups.append(trajs)
        pair_groups.append(pairs)
        all_pairs.extend(pairs)

        makespans = [t.makespan for t in trajs]
        summary.append({
            "instance_id": instance_id,
            "num_trajectories": len(trajs),
            "num_pairs": len(pairs),
            "best_makespan": min(makespans),
            "worst_makespan": max(makespans),
            "avg_makespan": sum(makespans) / len(makespans),
        })

    return {
        "trajectory_groups": trajectory_groups,
        "pair_groups": pair_groups,
        "all_pairs": all_pairs,
        "summary": summary,
    }