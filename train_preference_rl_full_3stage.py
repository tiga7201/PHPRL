import random
import numpy as np
import torch

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv

from rl.replay_buffer_pref import ReplayBufferPref
from rl.diversity_collector import collect_diversity_data_for_env
from rl.reward_learning_step import run_reward_learning_step
from rl.sac_agent import SACAgent

from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from models.shypergnn_full import SHyperGNNFull


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


def collect_one_episode(env, agent, greedy=False):
    from utils.graph_builder import build_hypergraph_state

    env.reset()
    done = False
    total_reward = 0.0
    last_info = None

    while not done:
        graph_state = build_hypergraph_state(env)
        action, edge_idx = agent.select_action(graph_state, greedy=greedy)
        _, reward, done, info = env.step(action)
        total_reward += reward
        last_info = info

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
        _, makespan = collect_one_episode(env, agent, greedy=True)
        makespans.append(makespan)
    return sum(makespans) / len(makespans), makespans


def train_preference_rl_full_3stage(
    # Step 1
    I1=5,
    # Step 3
    I2=20,
    batch_instance_size=4,
    num_trajectories_per_instance=5,
    k_neighbors=3,
    reward_model_epochs=5,
    updates_per_iter=4,
    instance_refresh_interval=5,
    buffer_capacity=50000,
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
):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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

    encoder = SHyperGNNFull(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    replay_buffer = ReplayBufferPref(capacity=buffer_capacity)

    eval_seeds = [100, 101, 102, 103, 104]
    best_eval_avg = float("inf")

    next_seed = 0
    reward_model = None
    reward_optimizer = None

    # -------------------------------
    # Step 1: Diversity exploration
    # -------------------------------
    print("\n=== Step 1: Diversity exploration ===")
    for iteration in range(I1):
        for _ in range(batch_instance_size):
            env = make_env(
                seed=next_seed,
                num_jobs=num_jobs,
                num_machines=num_machines,
                num_workers=num_workers,
                min_ops_per_job=min_ops_per_job,
                max_ops_per_job=max_ops_per_job,
            )
            instance_id = f"inst_{next_seed}"
            next_seed += 1

            collect_diversity_data_for_env(
                env=env,
                agent=agent,
                encoder=encoder,
                replay_buffer=replay_buffer,
                instance_id=instance_id,
                num_trajectories=num_trajectories_per_instance,
                k_neighbors=k_neighbors,
            )

        if len(replay_buffer) >= batch_size:
            stats_list = []
            for _ in range(updates_per_iter):
                batch = replay_buffer.sample(batch_size, reward_key="reward_div")
                stat = agent.update(batch)
                stats_list.append(stat)

            avg_q1 = sum(x["q1_loss"] for x in stats_list) / len(stats_list)
            avg_q2 = sum(x["q2_loss"] for x in stats_list) / len(stats_list)
            avg_actor = sum(x["actor_loss"] for x in stats_list) / len(stats_list)
        else:
            avg_q1 = avg_q2 = avg_actor = None

        eval_avg, eval_list = evaluate_on_fixed_seeds(
            agent,
            eval_seeds,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
        )

        print(
            f"[Step1] Iter {iteration + 1:03d} | "
            f"buffer={len(replay_buffer)} | "
            f"eval_avg={eval_avg:.4f}"
        )
        if avg_q1 is not None:
            print(
                f"  q1_loss={avg_q1:.4f} | q2_loss={avg_q2:.4f} | actor_loss={avg_actor:.4f}"
            )
        print("  eval_makespans:", [round(x, 4) for x in eval_list])

    # -------------------------------
    # Step 2: Reward learning
    # -------------------------------
    print("\n=== Step 2: Reward learning ===")
    reward_model, reward_optimizer, summary = run_reward_learning_step(
        replay_buffer=replay_buffer,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        lr=lr,
        min_gap=0.0,
        reward_model=reward_model,
        optimizer=reward_optimizer,
        num_epochs=reward_model_epochs,
    )
    print("Reward learning summary:", summary)

    # -------------------------------
    # Step 3: Policy learning with reward_obj
    # -------------------------------
    print("\n=== Step 3: Policy learning with reward_obj ===")
    current_batch_seeds = []

    for iteration in range(I2):
        if (iteration % instance_refresh_interval == 0) or (len(current_batch_seeds) == 0):
            current_batch_seeds = list(range(next_seed, next_seed + batch_instance_size))
            next_seed += batch_instance_size

        # collect new trajectories for current batch, still needed to enrich buffer
        for seed in current_batch_seeds:
            env = make_env(
                seed=seed,
                num_jobs=num_jobs,
                num_machines=num_machines,
                num_workers=num_workers,
                min_ops_per_job=min_ops_per_job,
                max_ops_per_job=max_ops_per_job,
            )
            instance_id = f"inst_{seed}"

            collect_diversity_data_for_env(
                env=env,
                agent=agent,
                encoder=encoder,
                replay_buffer=replay_buffer,
                instance_id=instance_id,
                num_trajectories=num_trajectories_per_instance,
                k_neighbors=k_neighbors,
            )

        # periodically re-train reward model and relabel buffer
        if iteration % instance_refresh_interval == 0:
            reward_model, reward_optimizer, summary = run_reward_learning_step(
                replay_buffer=replay_buffer,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                lr=lr,
                min_gap=0.0,
                reward_model=reward_model,
                optimizer=reward_optimizer,
                num_epochs=reward_model_epochs,
            )
        else:
            summary = {"num_pairs": 0, "reward_loss": 0.0, "reward_acc": 0.0}

        stats_list = []
        if len(replay_buffer) >= batch_size:
            for _ in range(updates_per_iter):
                batch = replay_buffer.sample(batch_size, reward_key="reward_obj")
                stat = agent.update(batch)
                stats_list.append(stat)

            avg_q1 = sum(x["q1_loss"] for x in stats_list) / len(stats_list)
            avg_q2 = sum(x["q2_loss"] for x in stats_list) / len(stats_list)
            avg_actor = sum(x["actor_loss"] for x in stats_list) / len(stats_list)
        else:
            avg_q1 = avg_q2 = avg_actor = None

        eval_avg, eval_list = evaluate_on_fixed_seeds(
            agent,
            eval_seeds,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
        )

        if eval_avg < best_eval_avg:
            best_eval_avg = eval_avg
            torch.save(agent.actor.state_dict(), "best_pref3stage_actor.pt")
            torch.save(agent.q1.state_dict(), "best_pref3stage_q1.pt")
            torch.save(agent.q2.state_dict(), "best_pref3stage_q2.pt")
            if reward_model is not None:
                torch.save(reward_model.state_dict(), "best_pref3stage_reward_model.pt")

        print(
            f"[Step3] Iter {iteration + 1:03d} | "
            f"buffer={len(replay_buffer)} | "
            f"eval_avg={eval_avg:.4f} | "
            f"best_eval_avg={best_eval_avg:.4f} | "
            f"reward_pairs={summary['num_pairs']} | "
            f"reward_loss={summary['reward_loss']:.4f} | "
            f"reward_acc={summary['reward_acc']:.4f}"
        )
        if avg_q1 is not None:
            print(
                f"  q1_loss={avg_q1:.4f} | q2_loss={avg_q2:.4f} | actor_loss={avg_actor:.4f}"
            )
        print("  train_batch_seeds:", current_batch_seeds)
        print("  eval_makespans:", [round(x, 4) for x in eval_list])

    return agent, reward_model


if __name__ == "__main__":
    train_preference_rl_full_3stage()