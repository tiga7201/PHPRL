import random
import numpy as np
import torch

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from rl.replay_buffer import ReplayBuffer
from rl.sac_agent import SACAgent
from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull


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
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_workers=num_workers,
        min_ops_per_job=min_ops_per_job,
        max_ops_per_job=max_ops_per_job,
        seed=seed,
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


def build_training_batch_envs(
    start_seed,
    batch_instance_size,
    num_jobs=3,
    num_machines=3,
    num_workers=3,
    min_ops_per_job=2,
    max_ops_per_job=4,
):
    envs = []
    seeds = []

    for i in range(batch_instance_size):
        seed = start_seed + i
        env = make_env(
            seed=seed,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
        )
        envs.append(env)
        seeds.append(seed)

    return envs, seeds


def train_sac_shyper_full_fixedscale_batch(
    num_iterations=20,
    batch_instance_size=8,
    episodes_per_instance_batch=5,
    updates_per_episode_round=2,
    instance_refresh_interval=5,
    buffer_capacity=30000,
    batch_size=16,
    hidden_dim=64,
    num_layers=2,
    lr=1e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.1,
    num_jobs=3,
    num_machines=3,
    num_workers=3,
    min_ops_per_job=2,
    max_ops_per_job=4,
    return_history=False,
):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(
        f"FULL SHyper | layers={num_layers} | Fixed scale: "
        f"jobs={num_jobs}, machines={num_machines}, workers={num_workers}, "
        f"ops/job in [{min_ops_per_job}, {max_ops_per_job}]"
    )

    actor = SHyperActorFull(hidden_dim=hidden_dim, num_layers=num_layers)
    q1 = SHyperQCriticFull(hidden_dim=hidden_dim, num_layers=num_layers)
    q2 = SHyperQCriticFull(hidden_dim=hidden_dim, num_layers=num_layers)

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

    next_train_seed = 0
    batch_envs = None
    batch_env_seeds = None

    history = []
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
                    stat = agent.update(batch)
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
            torch.save(agent.actor.state_dict(), "best_fixedscale_full_sac_actor.pt")
            torch.save(agent.q1.state_dict(), "best_fixedscale_full_sac_q1.pt")
            torch.save(agent.q2.state_dict(), "best_fixedscale_full_sac_q2.pt")

        train_reward_avg = sum(train_rewards_this_iter) / len(train_rewards_this_iter)
        train_makespan_avg = sum(train_makespans_this_iter) / len(train_makespans_this_iter)

        history.append({
            "iteration": iteration + 1,
            "train_reward_avg": float(train_reward_avg),
            "train_makespan_avg": float(train_makespan_avg),
            "eval_avg": float(eval_avg),
            "best_eval_avg": float(best_eval_avg),
            "buffer": int(len(buffer)),
            "q1_loss": None if avg_q1_loss is None else float(avg_q1_loss),
            "q2_loss": None if avg_q2_loss is None else float(avg_q2_loss),
            "actor_loss": None if avg_actor_loss is None else float(avg_actor_loss),
            "eval_makespans": [float(x) for x in eval_list],
            "train_batch_seeds": [int(x) for x in batch_env_seeds],
        })

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
            print("  current_train_batch_seeds:", batch_env_seeds)
            print("  eval_makespans:", [round(x, 4) for x in eval_list])

    if return_history:
        meta = {
            "best_checkpoint_actor": "best_fixedscale_full_sac_actor.pt",
            "best_checkpoint_q1": "best_fixedscale_full_sac_q1.pt",
            "best_checkpoint_q2": "best_fixedscale_full_sac_q2.pt",
            "best_eval_avg": float(best_eval_avg),
        }
        return agent, history, meta

    return agent


if __name__ == "__main__":
    train_sac_shyper_full_fixedscale_batch(
        num_iterations=50,
        batch_instance_size=8,
        episodes_per_instance_batch=5,
        updates_per_episode_round=2,
        instance_refresh_interval=5,
        hidden_dim=64,
        num_layers=2,
        num_jobs=3,
        num_machines=3,
        num_workers=3,
        min_ops_per_job=2,
        max_ops_per_job=4,
    )