import torch


def relabel_trajectory_groups_with_reward_model(trajectory_groups, reward_model, batch_size=64):
    """
    Use cached sa_embeddings to relabel reward_obj for all transitions.
    """
    device = next(reward_model.parameters()).device

    all_sa_embs = []
    all_transition_refs = []

    for trajs in trajectory_groups:
        for traj in trajs:
            if traj.sa_embeddings is None or traj.transition_refs is None:
                continue

            for sa_emb, transition in zip(traj.sa_embeddings, traj.transition_refs):
                all_sa_embs.append(sa_emb)
                all_transition_refs.append(transition)

    if len(all_sa_embs) == 0:
        return 0

    num_relabeled = 0

    for start in range(0, len(all_sa_embs), batch_size):
        batch_embs = all_sa_embs[start:start + batch_size]
        batch_refs = all_transition_refs[start:start + batch_size]

        batch_tensor = torch.stack([
            emb if torch.is_tensor(emb) else torch.as_tensor(emb, dtype=torch.float32)
            for emb in batch_embs
        ], dim=0).to(device)

        with torch.no_grad():
            scores = reward_model.step_head(batch_tensor).squeeze(-1)

        for score, transition in zip(scores, batch_refs):
            transition["reward_obj"] = float(score.item())
            num_relabeled += 1

    return num_relabeled