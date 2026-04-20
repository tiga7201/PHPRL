import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl.preference_batch_collector import (
    build_training_batch_envs,
    collect_preference_data_for_batch,
)
from rl.sac_agent import SACAgent
from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from models.reward_model import RewardModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_pair_loss_and_accuracy(reward_model, pairs):
    losses = []
    correct = 0

    for pair in pairs:
        score_better = reward_model.score_trajectory(pair.traj_better)
        score_worse = reward_model.score_trajectory(pair.traj_worse)

        prob = torch.sigmoid(score_better - score_worse)
        target = torch.tensor(1.0, device=prob.device)

        loss = F.binary_cross_entropy(prob, target)
        losses.append(loss)

        if prob.item() > 0.5:
            correct += 1

    loss = torch.stack(losses).mean()
    acc = correct / len(pairs) if len(pairs) > 0 else 0.0
    return loss, acc


def train_reward_model_from_preferences(
    num_epochs=50,
    batch_instance_size=4,
    num_trajectories_per_instance=6,
    min_gap=0.0,
    hidden_dim=64,
    num_layers=2,
    lr=1e-4,
    num_jobs=3,
    num_machines=3,
    num_workers=3,
    min_ops_per_job=2,
    max_ops_per_job=4,
):
    set_seed(42)

    # 用一个 full SHyper SAC agent 来产生轨迹
    actor = SHyperActorFull(hidden_dim=hidden_dim, num_layers=num_layers)
    q1 = SHyperQCriticFull(hidden_dim=hidden_dim, num_layers=num_layers)
    q2 = SHyperQCriticFull(hidden_dim=hidden_dim, num_layers=num_layers)

    agent = SACAgent(
        actor=actor,
        q1=q1,
        q2=q2,
        lr=1e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        device=device,
    )

    # 固定一批实例，先收集 preference 数据
    envs, instance_ids = build_training_batch_envs(
        start_seed=0,
        batch_instance_size=batch_instance_size,
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_workers=num_workers,
        min_ops_per_job=min_ops_per_job,
        max_ops_per_job=max_ops_per_job,
    )

    data = collect_preference_data_for_batch(
        envs=envs,
        instance_ids=instance_ids,
        agent=agent,
        num_trajectories_per_instance=num_trajectories_per_instance,
        min_gap=min_gap,
    )

    pairs = data["all_pairs"]
    print("Total preference pairs:", len(pairs))

    reward_model = RewardModel(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = optim.Adam(reward_model.parameters(), lr=lr)

    best_acc = 0.0

    for epoch in range(num_epochs):
        loss, acc = compute_pair_loss_and_accuracy(reward_model, pairs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        best_acc = max(best_acc, acc)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"loss={loss.item():.4f} | "
                f"pair_acc={acc:.4f} | "
                f"best_acc={best_acc:.4f}"
            )

    return reward_model, data


if __name__ == "__main__":
    train_reward_model_from_preferences()