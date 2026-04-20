from rl.trajectory_collector import Trajectory


def trajectory_from_dict(d):
    return Trajectory(
        instance_id=d["instance_id"],
        graph_states=d["graph_states"],
        edge_indices=d["edge_indices"],
        actions=d["actions"],
        rewards=d["rewards"],
        makespan=d["makespan"],
        sa_embeddings=None,
        transition_refs=None,
    )


def trajectories_from_dicts(dicts):
    return [trajectory_from_dict(d) for d in dicts]


def group_trajectories_by_instance_id(trajectories):
    groups = {}
    for traj in trajectories:
        groups.setdefault(traj.instance_id, []).append(traj)
    return list(groups.values())