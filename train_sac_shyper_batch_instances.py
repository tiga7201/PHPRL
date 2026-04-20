import random
import numpy as np
import torch

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from rl.replay_buffer import ReplayBuffer
from rl.sac_agent import SACAgent
from models.actor_shyper import SHyperActor
from models.q_critic_shyper import SHyperQCritic


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed=None):
    instance = generate_random_instance(seed=seed)
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


def evaluate_on_fixed_seeds(agent, eval_seeds):
    makespans = []
    for seed in eval_seeds:
        env = make_env(seed=seed)
        _, makespan = collect_one_episode(env, agent, buffer=None, greedy=True)
        makespans.append(makespan)
    return sum(makespans) / len(makespans), makespans


def train_sac_shyper_batch_instances(
    num_iterations=50,
    batch_instance_size=5,
    episodes_per_instance_batch=5,
    updates_per_episode_round=4,
    buffer_capacity=20000,
    batch_size=16,
    hidden_dim=64,
    lr=1e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.1,
):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    actor = SHyperActor(hidden_dim=hidden_dim)
    q1 = SHyperQCritic(hidden_dim=hidden_dim)
    q2 = SHyperQCritic(hidden_dim=hidden_dim)

    agent = SACAgent(
        actor=actor,
        q1=q1,
        q2=q2,
        lr=lr,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        device=device,
    )

    buffer = ReplayBuffer(capacity=buffer_capacity)

    eval_seeds = [100, 101, 102, 103, 104]
    best_eval_avg = float("inf")

    global_instance_seed = 0

    for iteration in range(num_iterations):
        # fixed batch of training instances for this iteration
        batch_envs = []
        batch_env_seeds = []
        for _ in range(batch_instance_size):
            env_seed = global_instance_seed
            global_instance_seed += 1
            batch_envs.append(make_env(seed=env_seed))
            batch_env_seeds.append(env_seed)

        train_rewards_this_iter = []
        train_makespans_this_iter = []

        for episode_round in range(episodes_per_instance_batch):
            # solve each instance once under current policy
            for env in batch_envs:
                total_reward, makespan = collect_one_episode(
                    env, agent, buffer=buffer, greedy=False
                )
                train_rewards_this_iter.append(total_reward)
                train_makespans_this_iter.append(makespan)

            # do updates after one full round over the batch
            if len(buffer) >= batch_size:
                update_stats = []
                for _ in range(updates_per_episode_round):
                    batch = buffer.sample(batch_size)
                    stat = agent.update(batch)
                    update_stats.append(stat)

                avg_q1_loss = sum(s["q1_loss"] for s in update_stats) / len(update_stats)
                avg_q2_loss = sum(s["q2_loss"] for s in update_stats) / len(update_stats)
                avg_actor_loss = sum(s["actor_loss"] for s in update_stats) / len(update_stats)
            else:
                avg_q1_loss = None
                avg_q2_loss = None
                avg_actor_loss = None

        eval_avg, eval_list = evaluate_on_fixed_seeds(agent, eval_seeds)

        if eval_avg < best_eval_avg:
            best_eval_avg = eval_avg
            torch.save(agent.actor.state_dict(), "best_batch_sac_shyper_actor.pt")
            torch.save(agent.q1.state_dict(), "best_batch_sac_shyper_q1.pt")
            torch.save(agent.q2.state_dict(), "best_batch_sac_shyper_q2.pt")

        train_reward_avg = sum(train_rewards_this_iter) / len(train_rewards_this_iter)
        train_makespan_avg = sum(train_makespans_this_iter) / len(train_makespans_this_iter)

        if avg_q1_loss is None:
            print(
                f"Iteration {iteration + 1:03d} | "
                f"train_reward_avg={train_reward_avg:.4f} | "
                f"train_makespan_avg={train_makespan_avg:.4f} | "
                f"eval_avg={eval_avg:.4f} | "
                f"best_eval_avg={best_eval_avg:.4f} | "
                f"buffer={len(buffer)}"
            )
        else:
            print(
                f"Iteration {iteration + 1:03d} | "
                f"train_reward_avg={train_reward_avg:.4f} | "
                f"train_makespan_avg={train_makespan_avg:.4f} | "
                f"eval_avg={eval_avg:.4f} | "
                f"best_eval_avg={best_eval_avg:.4f} | "
                f"buffer={len(buffer)} | "
                f"q1_loss={avg_q1_loss:.4f} | "
                f"q2_loss={avg_q2_loss:.4f} | "
                f"actor_loss={avg_actor_loss:.4f}"
            )
            print("  batch_env_seeds:", batch_env_seeds)
            print("  eval_makespans:", [round(x, 4) for x in eval_list])

    return agent


if __name__ == "__main__":
    train_sac_shyper_batch_instances(
        num_iterations=20,
        batch_instance_size=5,
        episodes_per_instance_batch=5,
        updates_per_episode_round=2,
    )