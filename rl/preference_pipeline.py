from collections import defaultdict

from rl.trajectory_collector import Trajectory
from rl.preference_dataset import build_preference_pairs_grouped


def rebuild_trajectories_from_buffer(replay_buffer):
    """
    Rebuild trajectories from ReplayBufferPref using:
      - instance_id
      - traj_id
      - step_id

    Returns:
      trajectory_groups: List[List[Trajectory]]
        grouped by instance
    """
    items = replay_buffer.get_all()

    # group transitions by (instance_id, traj_id)
    traj_map = defaultdict(list)
    for item in items:
        key = (item["instance_id"], item["traj_id"])
        traj_map[key].append(item)

    # rebuild trajectories
    instance_group_map = defaultdict(list)

    for (instance_id, traj_id), transitions in traj_map.items():
        transitions = sorted(transitions, key=lambda x: x["step_id"])

        graph_states = [x["state"] for x in transitions]
        edge_indices = [x["edge_idx"] for x in transitions]
        actions = [x["action"] for x in transitions]
        rewards = [x["reward_div"] for x in transitions]

        # recover a makespan-like trajectory quality proxy:
        # during Step 2 preference construction we still need the true trajectory quality.
        # For now we store it via sum of original diversity rewards? No:
        # better to attach env quality later if available.
        #
        # Here we infer from transition state is not possible.
        # So during Step 1 collection we should store makespan per traj into each transition.
        #
        # For compatibility, use traj-level field if present, otherwise fallback to sum rewards.
        if "traj_makespan" in transitions[0]:
            makespan = float(transitions[0]["traj_makespan"])
        else:
            makespan = float(sum(rewards))

        transition_refs = transitions
        traj = Trajectory(
            instance_id=instance_id,
            graph_states=graph_states,
            edge_indices=edge_indices,
            actions=actions,
            rewards=rewards,
            makespan=makespan,
            transition_refs=transition_refs,
        )
        instance_group_map[instance_id].append(traj)

    trajectory_groups = list(instance_group_map.values())
    return trajectory_groups


def build_preference_pairs_from_buffer(replay_buffer, min_gap=0.0):
    trajectory_groups = rebuild_trajectories_from_buffer(replay_buffer)
    pairs = build_preference_pairs_grouped(trajectory_groups, min_gap=min_gap)
    return trajectory_groups, pairs