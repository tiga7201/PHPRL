import random
import numpy as np
import torch

from env.instance_generator import create_demo_instance
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


def evaluate_greedy(env, agent, num_episodes=5):
    makespans = []
    for _ in range(num_episodes):
        _, makespan = collect_one_episode(env, agent, buffer=None, greedy=True)
        makespans.append(makespan)
    return sum(makespans) / len(makespans)


def train_sac_shyper(
    num_episodes=400,
    buffer_capacity=10000,
    batch_size=16,
    warmup_episodes=20,
    updates_per_episode=4,
    hidden_dim=64,
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.1,
):
    set_seed(42)

    env = FJSPWFEnv(create_demo_instance())

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
    )

    buffer = ReplayBuffer(capacity=buffer_capacity)
    best_greedy = float("inf")

    for episode in range(num_episodes):
        total_reward, makespan = collect_one_episode(env, agent, buffer=buffer, greedy=False)

        stats = None
        if episode + 1 >= warmup_episodes and len(buffer) >= batch_size:
            update_stats = []
            for _ in range(updates_per_episode):
                batch = buffer.sample(batch_size)
                stat = agent.update(batch)
                update_stats.append(stat)

            stats = {
                "q1_loss": sum(s["q1_loss"] for s in update_stats) / len(update_stats),
                "q2_loss": sum(s["q2_loss"] for s in update_stats) / len(update_stats),
                "actor_loss": sum(s["actor_loss"] for s in update_stats) / len(update_stats),
            }

        if (episode + 1) % 20 == 0:
            greedy_makespan = evaluate_greedy(env, agent, num_episodes=5)

            if greedy_makespan < best_greedy:
                best_greedy = greedy_makespan
                torch.save(agent.actor.state_dict(), "best_sac_shyper_actor.pt")
                torch.save(agent.q1.state_dict(), "best_sac_shyper_q1.pt")
                torch.save(agent.q2.state_dict(), "best_sac_shyper_q2.pt")

            if stats is None:
                print(
                    f"Episode {episode + 1:03d} | "
                    f"reward={total_reward:.4f} | "
                    f"makespan={makespan:.4f} | "
                    f"greedy={greedy_makespan:.4f} | "
                    f"best_greedy={best_greedy:.4f} | "
                    f"buffer={len(buffer)}"
                )
            else:
                print(
                    f"Episode {episode + 1:03d} | "
                    f"reward={total_reward:.4f} | "
                    f"makespan={makespan:.4f} | "
                    f"greedy={greedy_makespan:.4f} | "
                    f"best_greedy={best_greedy:.4f} | "
                    f"buffer={len(buffer)} | "
                    f"q1_loss={stats['q1_loss']:.4f} | "
                    f"q2_loss={stats['q2_loss']:.4f} | "
                    f"actor_loss={stats['actor_loss']:.4f}"
                )

    return agent


if __name__ == "__main__":
    train_sac_shyper()