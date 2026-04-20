import torch


def extract_state_action_embedding(encoder, graph_state, chosen_edge_idx: int):
    """
    Build the same state-action representation used by actor/critic heads:
      [op_emb, machine_emb, worker_emb, edge_emb, graph_emb]
    for the chosen edge only.
    """
    device = next(encoder.parameters()).device

    enc = encoder(graph_state)
    edge_links = torch.as_tensor(graph_state["edge_links"], dtype=torch.long, device=device)

    edge_idx = int(chosen_edge_idx)
    op_idx = int(edge_links[edge_idx, 0].item())
    machine_idx = int(edge_links[edge_idx, 1].item())
    worker_idx = int(edge_links[edge_idx, 2].item())

    op_emb = enc["op_emb"][op_idx]
    machine_emb = enc["machine_emb"][machine_idx]
    worker_emb = enc["worker_emb"][worker_idx]
    edge_emb = enc["edge_emb"][edge_idx]
    graph_emb = enc["graph_emb"]

    sa_emb = torch.cat(
        [op_emb, machine_emb, worker_emb, edge_emb, graph_emb],
        dim=-1,
    )  # [7d]

    return sa_emb


def cache_sa_embeddings_for_trajectory(trajectory, encoder):
    """
    Compute and attach step-level state-action embeddings for one trajectory.
    """
    sa_embeddings = []
    with torch.no_grad():
        for graph_state, edge_idx in zip(trajectory.graph_states, trajectory.edge_indices):
            sa_emb = extract_state_action_embedding(encoder, graph_state, edge_idx)
            sa_embeddings.append(sa_emb.detach().cpu())

    trajectory.sa_embeddings = sa_embeddings
    return trajectory


def cache_sa_embeddings_for_trajectory_groups(trajectory_groups, encoder):
    """
    Cache state-action embeddings for all trajectories in all groups.
    """
    for trajs in trajectory_groups:
        for traj in trajs:
            cache_sa_embeddings_for_trajectory(traj, encoder)
    return trajectory_groups