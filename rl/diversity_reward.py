import math
import torch


def get_graph_embedding(encoder, graph_state):
    """
    Use the encoder to extract graph-level embedding h_t.
    """
    with torch.no_grad():
        out = encoder(graph_state)
        h = out["graph_emb"]  # [3d]
    return h


def compute_diversity_rewards_for_trajectories(
    trajectories,
    encoder,
    k_neighbors=3,
):
    """
    Compute r_t^div for a list of trajectories of the SAME instance.

    Paper-aligned idea:
      - compare states at the same decision step t
      - use graph embeddings h_t
      - r_t^div = log( average distance to K nearest neighbor states )

    If there are no other states available at the same step, reward is 0.

    Returns:
      rewards_per_traj: List[List[float]]
        same shape as trajectories, one diversity reward per step
    """
    # precompute graph embeddings for all trajectories and all steps
    traj_embeddings = []
    for traj in trajectories:
        embs = []
        for graph_state in traj.graph_states:
            h = get_graph_embedding(encoder, graph_state)
            embs.append(h)
        traj_embeddings.append(embs)

    rewards_per_traj = []

    for i, traj in enumerate(trajectories):
        step_rewards = []

        for t, h_t in enumerate(traj_embeddings[i]):
            candidate_distances = []

            # compare only with other trajectories, same step t
            for j, other_traj in enumerate(trajectories):
                if i == j:
                    continue
                if t >= len(traj_embeddings[j]):
                    continue

                h_other = traj_embeddings[j][t]
                dist = torch.norm(h_t - h_other, p=2).item()
                candidate_distances.append(dist)

            # if no historical neighbors exist, reward = 0
            if len(candidate_distances) == 0:
                step_rewards.append(0.0)
                continue

            candidate_distances.sort()
            k = min(k_neighbors, len(candidate_distances))
            avg_dist = sum(candidate_distances[:k]) / k

            # If the same-step states across trajectories are nearly identical,
            # treat this as no diversity gain instead of giving a huge negative reward.
            eps = 1e-8
            if avg_dist < eps:
                r_div = 0.0
            else:
                r_div = math.log(avg_dist)

            step_rewards.append(float(r_div))

        rewards_per_traj.append(step_rewards)

    return rewards_per_traj