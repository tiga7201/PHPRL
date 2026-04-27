import torch
import torch.optim as optim
import random

from rl.preference_pipeline import build_preference_pairs_from_buffer
from rl.preference_trainer import train_reward_model_on_pairs
from models.reward_model import RewardModel
from rl.embedding_cache import cache_sa_embeddings_for_trajectory_groups
from rl.relabel_utils import relabel_trajectory_groups_with_reward_model


def relabel_replay_buffer_with_reward_model(replay_buffer, reward_model):
    """
    Recompute reward_obj for every transition in replay buffer.
    """
    for item in replay_buffer.get_all():
        state = item["state"]
        edge_idx = item["edge_idx"]

        with torch.no_grad():
            score = reward_model.score_state_action(state, edge_idx)

        item["reward_obj"] = float(score.item())


def run_reward_learning_step(
    replay_buffer,
    hidden_dim=64,
    num_layers=2,
    lr=1e-4,
    min_gap=0.0,
    reward_model=None,
    optimizer=None,
    num_epochs=10,
    max_preference_pairs=1000,
):
    """
    Step 2:
      1) rebuild trajectories from buffer
      2) build preference pairs within each instance
      3) train reward model
      4) relabel replay buffer with reward_obj
    """
    trajectory_groups, pairs = build_preference_pairs_from_buffer(
        replay_buffer=replay_buffer,
        min_gap=min_gap,
    )
    if (max_preference_pairs is not None) and (len(pairs) > max_preference_pairs):
        pairs = random.sample(pairs, max_preference_pairs)

    created_new_model = False
    if reward_model is None:
        reward_model = RewardModel(hidden_dim=hidden_dim, num_layers=num_layers)
        optimizer = optim.Adam(reward_model.parameters(), lr=lr)
        created_new_model = True

    if optimizer is None:
        optimizer = optim.Adam(reward_model.parameters(), lr=lr)

    # cache state-action embeddings for faster reward learning
    cache_encoder = reward_model.encoder
    cache_sa_embeddings_for_trajectory_groups(trajectory_groups, cache_encoder)

    stats = {"loss": 0.0, "acc": 0.0}
    num_relabeled = 0
    if len(pairs) > 0:
        stats = train_reward_model_on_pairs(
            reward_model=reward_model,
            pairs=pairs,
            optimizer=optimizer,
            num_epochs=num_epochs,
            pair_batch_size=16,
        )
        # relabel_replay_buffer_with_reward_model(replay_buffer, reward_model)
        num_relabeled = relabel_trajectory_groups_with_reward_model(
            trajectory_groups=trajectory_groups,
            reward_model=reward_model,
            batch_size=64,
        )

    summary = {
        "num_instances": len(trajectory_groups),
        "num_pairs": len(pairs),
        "reward_loss": stats["loss"],
        "reward_acc": stats["acc"],
        "created_new_model": created_new_model,
        "num_relabeled": num_relabeled,
    }

    return reward_model, optimizer, summary