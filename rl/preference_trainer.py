import random
import torch
import torch.nn.functional as F


def iterate_pair_minibatches(pairs, batch_size, shuffle=True):
    indices = list(range(len(pairs)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield [pairs[i] for i in batch_idx]


def compute_pair_loss_and_accuracy_batch(reward_model, pair_batch):
    better_trajs = [pair.traj_better for pair in pair_batch]
    worse_trajs = [pair.traj_worse for pair in pair_batch]

    score_better = reward_model.score_trajectory_batch(better_trajs)   # [B]
    score_worse = reward_model.score_trajectory_batch(worse_trajs)     # [B]

    prob = torch.sigmoid(score_better - score_worse)                   # [B]
    target = torch.ones_like(prob)

    loss = F.binary_cross_entropy(prob, target)

    acc = ((prob > 0.5).float().mean()).item()
    return loss, acc


def train_reward_model_on_pairs(
    reward_model,
    pairs,
    optimizer,
    num_epochs=5,
    pair_batch_size=16,
):
    if len(pairs) == 0:
        return {"loss": 0.0, "acc": 0.0}

    history = []

    for _ in range(num_epochs):
        epoch_losses = []
        epoch_accs = []

        for pair_batch in iterate_pair_minibatches(
            pairs,
            batch_size=pair_batch_size,
            shuffle=True,
        ):
            loss, acc = compute_pair_loss_and_accuracy_batch(reward_model, pair_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_accs.append(acc)

        history.append({
            "loss": sum(epoch_losses) / len(epoch_losses),
            "acc": sum(epoch_accs) / len(epoch_accs),
        })

    return history[-1]