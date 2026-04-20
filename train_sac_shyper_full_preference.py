import random
import numpy as np
import torch
import torch.optim as optim

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from rl.replay_buffer import ReplayBuffer
from rl.preference_batch_collector import (
    build_training_batch_envs,
    collect_preference_data_for_batch,
)
from rl.preference_trainer import train_reward_model_on_pairs
from rl.sac_agent_preference import SACAgentPreference

from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from models.reward_model import RewardModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(
    seed=None,
    num_jobs=3,
    num_machines=3,
    num_workers=3,
    min_ops_per_job=2,
    max_ops_per_job=4,
):
    instance = generate_random_instance(
        seed=seed,
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_workers=num_workers,
        min_ops_per_job=min_ops_per_job,
        max_ops_per_job=max_ops_per_job,
    )
    return FJSPWFEnv(instance)


def collect_one_episode(env, agent, buffer=None, greedy=False):
    state = env.reset()
    done = False
    total_reward = 0.0
    last_info = None

    while not done:
        graph_state = build_hypergraph_state(env)
        action, edge_idx = agent.select_action(graph_state, greedy=greedy)

        next_state, reward, done, info = env.step(action)
        next_graph_state = build_hypergraph_state(env)

        if (not greedy) and (buffer is not None):
            buffer.add(
                state=graph_state,
                action=action,
                edge_idx=edge_idx,
                reward=reward,
                next_state=next_graph_state,
                done=done,
            )

        total_reward += reward
        last_info = info
        state = next_state

    return total_reward, last_info["makespan"]


def evaluate_on_fixed_seeds(
    agent,
    eval_seeds,
    num_jobs=3,
    num_machines=3,
    num_workers=3,
    min_ops_per_job=2,
    max_ops_per_job=4,
):
    makespans = []
    for seed in eval_seeds:
        env = make_env(
            seed=seed,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
        )
        _, makespan = collect_one_episode(env, agent, buffer=None, greedy=True)
        makespans.append(makespan)
    return sum(makespans) / len(makespans), makespans


def train_sac_shyper_full_preference(
    num_iterations=20,
    batch_instance_size=4,
    episodes_per_instance_batch=5,
    preference_trajectories_per_instance=6,
    updates_per_episode_round=2,
    instance_refresh_interval=5,
    reward_model_epochs_per_iter=3,
    buffer_capacity=30000,
    batch_size=16,
    hidden_dim=64,
    num_layers=2,
    lr=1e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.1,
    pref_beta=0.1,
    num_jobs=3,
    num_machines=3,
    num_workers=3,
    min_ops_per_job=2,
    max_ops_per_job=4,
):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    actor = SHyperActorFull(hidden_dim=hidden_dim, num_layers=num_layers)
    q1 = SHyperQCriticFull(hidden_dim=hidden_dim, num_layers=num_layers)
    q2 = SHyperQCriticFull(hidden_dim=hidden_dim, num_layers=num_layers)

    agent = SACAgentPreference(
        actor=actor,
        q1=q1,
        q2=q2,
        lr=lr,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        pref_beta=pref_beta,
        device=device,
    )

    reward_model = RewardModel(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    reward_model_optimizer = optim.Adam(reward_model.parameters(), lr=lr)

    buffer = ReplayBuffer(capacity=buffer_capacity)

    eval_seeds = [100, 101, 102, 103, 104]
    best_eval_avg = float("inf")

    next_train_seed = 0
    batch_envs = None
    batch_env_seeds = None

    for iteration in range(num_iterations):
        if (iteration % instance_refresh_interval == 0) or (batch_envs is None):
            batch_envs, batch_env_seeds = build_training_batch_envs(
                start_seed=next_train_seed,
                batch_instance_size=batch_instance_size,
                num_jobs=num_jobs,
                num_machines=num_machines,
                num_workers=num_workers,
                min_ops_per_job=min_ops_per_job,
                max_ops_per_job=max_ops_per_job,
            )
            next_train_seed += batch_instance_size

        # ---- step 1: collect preference data on current batch ----
        pref_data = collect_preference_data_for_batch(
            envs=batch_envs,
            instance_ids=[f"inst_seed_{s}" for s in batch_env_seeds],
            agent=agent,
            num_trajectories_per_instance=preference_trajectories_per_instance,
            min_gap=0.0,
        )

        all_pairs = pref_data["all_pairs"]

        # ---- step 2: train reward model ----
        reward_stats = {"loss": 0.0, "acc": 0.0}
        if len(all_pairs) > 0:
            reward_stats = train_reward_model_on_pairs(
                reward_model=reward_model,
                pairs=all_pairs,
                optimizer=reward_model_optimizer,
                num_epochs=reward_model_epochs_per_iter,
            )

        # ---- step 3: standard repeated-solving RL data collection ----
        train_rewards_this_iter = []
        train_makespans_this_iter = []

        avg_q1_loss = None
        avg_q2_loss = None
        avg_actor_loss = None

        for episode_round in range(episodes_per_instance_batch):
            for env in batch_envs:
                total_reward, makespan = collect_one_episode(
                    env, agent, buffer=buffer, greedy=False
                )
                train_rewards_this_iter.append(total_reward)
                train_makespans_this_iter.append(makespan)

            if len(buffer) >= batch_size:
                update_stats = []
                for _ in range(updates_per_episode_round):
                    batch = buffer.sample(batch_size)
                    stat = agent.update(batch, reward_model=reward_model)
                    update_stats.append(stat)

                avg_q1_loss = sum(s["q1_loss"] for s in update_stats) / len(update_stats)
                avg_q2_loss = sum(s["q2_loss"] for s in update_stats) / len(update_stats)
                avg_actor_loss = sum(s["actor_loss"] for s in update_stats) / len(update_stats)

        eval_avg, eval_list = evaluate_on_fixed_seeds(
            agent=agent,
            eval_seeds=eval_seeds,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
        )

        if eval_avg < best_eval_avg:
            best_eval_avg = eval_avg
            torch.save(agent.actor.state_dict(), "best_preference_full_actor.pt")
            torch.save(agent.q1.state_dict(), "best_preference_full_q1.pt")
            torch.save(agent.q2.state_dict(), "best_preference_full_q2.pt")
            torch.save(reward_model.state_dict(), "best_preference_reward_model.pt")

        train_reward_avg = sum(train_rewards_this_iter) / len(train_rewards_this_iter)
        train_makespan_avg = sum(train_makespans_this_iter) / len(train_makespans_this_iter)

        print(
            f"Iteration {iteration + 1:03d} | "
            f"train_reward_avg={train_reward_avg:.4f} | "
            f"train_makespan_avg={train_makespan_avg:.4f} | "
            f"eval_avg={eval_avg:.4f} | "
            f"best_eval_avg={best_eval_avg:.4f} | "
            f"buffer={len(buffer)} | "
            f"pref_pairs={len(all_pairs)} | "
            f"reward_loss={reward_stats['loss']:.4f} | "
            f"reward_acc={reward_stats['acc']:.4f}"
        )

        if avg_q1_loss is not None:
            print(
                f"  q1_loss={avg_q1_loss:.4f} | "
                f"q2_loss={avg_q2_loss:.4f} | "
                f"actor_loss={avg_actor_loss:.4f}"
            )
        print("  current_train_batch_seeds:", batch_env_seeds)
        print("  eval_makespans:", [round(x, 4) for x in eval_list])

    return agent, reward_model


if __name__ == "__main__":
    train_sac_shyper_full_preference()