from rl.trajectory_collector import collect_multiple_trajectories_for_env
from rl.diversity_reward import compute_diversity_rewards_for_trajectories


def collect_diversity_data_for_env(
    env,
    agent,
    encoder,
    replay_buffer,
    instance_id: str,
    num_trajectories: int,
    k_neighbors: int = 3,
):
    """
    For one fixed instance:
      1) collect multiple trajectories
      2) compute diversity rewards r_t^div
      3) write all step transitions into ReplayBufferPref

    Returns:
      {
        "trajectories": trajectories,
        "div_rewards": rewards_per_traj,
        "summary": {...}
      }
    """
    trajectories = collect_multiple_trajectories_for_env(
        env=env,
        agent=agent,
        instance_id=instance_id,
        num_trajectories=num_trajectories,
    )

    rewards_per_traj = compute_diversity_rewards_for_trajectories(
        trajectories=trajectories,
        encoder=encoder,
        k_neighbors=k_neighbors,
    )

    # write transitions into replay buffer
    for traj_idx, (traj, div_rewards) in enumerate(zip(trajectories, rewards_per_traj)):
        num_steps = len(traj.graph_states)

        for step_id in range(num_steps):
            state = traj.graph_states[step_id]
            action = traj.actions[step_id]
            edge_idx = traj.edge_indices[step_id]
            reward_div = div_rewards[step_id]

            if step_id < num_steps - 1:
                next_state = traj.graph_states[step_id + 1]
                done = False
            else:
                next_state = None
                done = True

            replay_buffer.add(
                state=state,
                action=action,
                edge_idx=edge_idx,
                next_state=next_state,
                done=done,
                reward_div=reward_div,
                reward_obj=None,
                instance_id=instance_id,
                traj_id=f"{instance_id}_traj_{traj_idx}",
                step_id=step_id,
                traj_makespan=traj.makespan,
            )

    makespans = [traj.makespan for traj in trajectories]
    flat_div_rewards = [r for traj_rewards in rewards_per_traj for r in traj_rewards]

    summary = {
        "instance_id": instance_id,
        "num_trajectories": len(trajectories),
        "num_transitions": sum(len(traj.graph_states) for traj in trajectories),
        "best_makespan": min(makespans),
        "worst_makespan": max(makespans),
        "avg_makespan": sum(makespans) / len(makespans),
        "avg_div_reward": sum(flat_div_rewards) / len(flat_div_rewards) if flat_div_rewards else 0.0,
    }

    return {
        "trajectories": trajectories,
        "div_rewards": rewards_per_traj,
        "summary": summary,
    }