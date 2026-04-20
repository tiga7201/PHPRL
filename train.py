import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from models.actor import BaselineActor
from models.critic import BaselineCritic
from rl.replay_buffer import RolloutBuffer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_returns(rewards, gamma=1.0):
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.insert(0, g)
    return returns


def run_one_episode(env, actor, critic, buffer):
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        graph_state = build_hypergraph_state(env)

        value = critic(graph_state)

        decision = actor.sample_action(graph_state)
        action = decision["action"]
        edge_idx = decision["edge_idx"]
        log_prob = decision["log_prob"]

        next_state, reward, done, info = env.step(action)

        buffer.add(
            state=graph_state,
            action=action,
            edge_idx=edge_idx,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
        )

        total_reward += reward
        state = next_state

    return total_reward, info["makespan"]


def evaluate_greedy(env, actor):
    state = env.reset()
    done = False

    while not done:
        graph_state = build_hypergraph_state(env)
        decision = actor.select_greedy_action(graph_state)
        action = decision["action"]
        state, reward, done, info = env.step(action)

    return info["makespan"]


def train(num_episodes=300, lr=3e-4):
    set_seed(42)

    actor = BaselineActor()
    critic = BaselineCritic()

    actor_optim = optim.Adam(actor.parameters(), lr=lr)
    critic_optim = optim.Adam(critic.parameters(), lr=lr)

    env = FJSPWFEnv(create_demo_instance())

    best_makespan = float("inf")
    best_greedy = float("inf")
    for episode in range(num_episodes):
        buffer = RolloutBuffer()
        total_reward, makespan = run_one_episode(env, actor, critic, buffer)

        returns = compute_returns(buffer.rewards, gamma=1.0)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(buffer.values)
        log_probs = torch.stack(buffer.log_probs)

        advantages = returns - values.detach()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        best_makespan = min(best_makespan, makespan)

        if (episode + 1) % 20 == 0:
            greedy_makespan = evaluate_greedy(env, actor)
            print(
                f"Episode {episode + 1:03d} | "
                f"total_reward={total_reward:.4f} | "
                f"makespan={makespan:.4f} | "
                f"greedy={greedy_makespan:.4f} | "
                f"best_makespan={best_makespan:.4f} | "
                f"actor_loss={actor_loss.item():.4f} | "
                f"critic_loss={critic_loss.item():.4f}"
            )
            if greedy_makespan < best_greedy:
                best_greedy = greedy_makespan
                torch.save(actor.state_dict(), "best_actor.pt")
                torch.save(critic.state_dict(), "best_critic.pt")

    return actor, critic


if __name__ == "__main__":
    train()
