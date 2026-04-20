from rl.parallel_rollout import collect_trajectories_parallel
from rl.trajectory_utils import trajectories_from_dicts, group_trajectories_by_instance_id

from models.actor_shyper_full import SHyperActorFull


def main():
    actor = SHyperActorFull(hidden_dim=64, num_layers=2)

    seeds = [0, 1, 2, 3]

    traj_dicts = collect_trajectories_parallel(
        seeds=seeds,
        actor=actor,
        actor_class=SHyperActorFull,
        actor_kwargs={"hidden_dim": 64, "num_layers": 2},
        num_trajectories_per_instance=4,
        num_jobs=3,
        num_machines=3,
        num_workers=3,
        min_ops_per_job=2,
        max_ops_per_job=4,
        max_workers=4,
    )

    trajectories = trajectories_from_dicts(traj_dicts)
    groups = group_trajectories_by_instance_id(trajectories)

    print("Total trajectories:", len(trajectories))
    print("Total instance groups:", len(groups))

    for g in groups:
        print(
            g[0].instance_id,
            "num_traj =", len(g),
            "makespans =", [round(t.makespan, 4) for t in g]
        )


if __name__ == "__main__":
    main()