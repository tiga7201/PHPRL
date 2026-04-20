import torch
import torch.nn.functional as F
import torch.optim as optim

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from rl.sac_agent import SACAgent
from rl.trajectory_collector import collect_multiple_trajectories_for_env
from rl.preference_dataset import build_preference_pairs
from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from models.reward_model import RewardModel


def main():
    # build a full SHyper SAC agent (randomly initialized is enough for pipeline test)
    actor = SHyperActorFull(hidden_dim=64, num_layers=2)
    q1 = SHyperQCriticFull(hidden_dim=64, num_layers=2)
    q2 = SHyperQCriticFull(hidden_dim=64, num_layers=2)

    agent = SACAgent(
        actor=actor,
        q1=q1,
        q2=q2,
        lr=1e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        device="cpu",
    )

    # fixed one instance, collect multiple trajectories
    instance = generate_random_instance(
        seed=123,
        num_jobs=3,
        num_machines=3,
        num_workers=3,
        min_ops_per_job=2,
        max_ops_per_job=4,
    )
    env = FJSPWFEnv(instance)

    trajectories = collect_multiple_trajectories_for_env(
        env=env,
        agent=agent,
        instance_id="inst_123",
        num_trajectories=6,
    )

    print("Collected trajectories:", len(trajectories))
    print("Makespans:", [round(t.makespan, 4) for t in trajectories])

    pairs = build_preference_pairs(trajectories, min_gap=0.0)
    print("Preference pairs:", len(pairs))

    if len(pairs) == 0:
        print("No preference pairs built. Try increasing num_trajectories.")
        return

    reward_model = RewardModel(hidden_dim=64, num_layers=2)
    optimizer = optim.Adam(reward_model.parameters(), lr=1e-4)

    # take one training step on a few preference pairs
    batch_pairs = pairs[: min(4, len(pairs))]

    losses = []
    for pair in batch_pairs:
        score_better = reward_model.score_trajectory(pair.traj_better)
        score_worse = reward_model.score_trajectory(pair.traj_worse)

        # Bradley-Terry style probability
        prob = torch.sigmoid(score_better - score_worse)
        target = torch.tensor(1.0)

        loss = F.binary_cross_entropy(prob, target)
        losses.append(loss)

    total_loss = torch.stack(losses).mean()

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("Reward model train step OK.")
    print("Loss:", float(total_loss.item()))


if __name__ == "__main__":
    main()