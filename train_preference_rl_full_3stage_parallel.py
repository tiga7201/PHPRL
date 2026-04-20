import random
import numpy as np
import torch

from rl.parallel_rollout import collect_trajectories_parallel
from rl.trajectory_utils import trajectories_from_dicts, group_trajectories_by_instance_id
from rl.diversity_reward import compute_diversity_rewards_for_trajectories
from rl.replay_buffer_pref import ReplayBufferPref
from rl.reward_learning_step import run_reward_learning_step
from rl.sac_agent import SACAgent

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state

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

        env.reset()
        done = False
        last_info = None

        while not done:
            graph_state = build_hypergraph_state(env)
            decision = agent.actor.select_greedy_action(graph_state)
            action = decision["action"]
            _, _, done, info = env.step(action)
            last_info = info

        makespans.append(float(last_info["makespan"]))

    return sum(makespans) / len(makespans), makespans


def write_diversity_trajectories_to_buffer(
    trajectory_groups,
    encoder,
    replay_buffer,
    k_neighbors=3,
):
    """
    For each instance group:
      1) compute r_t^div
      2) write transitions into ReplayBufferPref
    """
    summaries = []

    for trajs in trajectory_groups:
        rewards_per_traj = compute_diversity_rewards_for_trajectories(
            trajectories=trajs,
            encoder=encoder,
            k_neighbors=k_neighbors,
        )

        for traj_idx, (traj, div_rewards) in enumerate(zip(trajs, rewards_per_traj)):
            num_steps = len(traj.graph_states)

            for step_id in range(num_steps):
                state = traj.graph_states[step_id]
                action = traj.actions[step_id]
                edge_idx = traj.edge_indices[step_id]
                reward_div = div_rewards[step_id]

                if step_id < num_steps - 1:
                    next_state = traj.graph_states[step_id + 1]
                    done = False
                else:
                    next_state = None
                    done = True

                replay_buffer.add(
                    state=state,
                    action=action,
                    edge_idx=edge_idx,
                    next_state=next_state,
                    done=done,
                    reward_div=reward_div,
                    reward_obj=None,
                    instance_id=traj.instance_id,
                    traj_id=f"{traj.instance_id}_traj_{traj_idx}",
                    step_id=step_id,
                    traj_makespan=traj.makespan,
                )

        makespans = [traj.makespan for traj in trajs]
        flat_div_rewards = [r for traj_rewards in rewards_per_traj for r in traj_rewards]

        summaries.append({
            "instance_id": trajs[0].instance_id,
            "num_trajectories": len(trajs),
            "num_transitions": sum(len(traj.graph_states) for traj in trajs),
            "best_makespan": min(makespans),
            "worst_makespan": max(makespans),
            "avg_makespan": sum(makespans) / len(makespans),
            "avg_div_reward": (
                sum(flat_div_rewards) / len(flat_div_rewards) if flat_div_rewards else 0.0
            ),
        })

    return summaries


def collect_parallel_diversity_batch(
    seeds,
    agent,
    encoder,
    replay_buffer,
    num_trajectories_per_instance,
    k_neighbors,
    num_jobs,
    num_machines,
    num_workers,
    min_ops_per_job,
    max_ops_per_job,
    max_workers,
    actor_class,
    actor_kwargs,
):
    traj_dicts = collect_trajectories_parallel(
        seeds=seeds,
        actor=agent.actor.cpu(),
        actor_class=actor_class,
        actor_kwargs=actor_kwargs,
        num_trajectories_per_instance=num_trajectories_per_instance,
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_workers=num_workers,
        min_ops_per_job=min_ops_per_job,
        max_ops_per_job=max_ops_per_job,
        max_workers=max_workers,
    )

    # move actor back to original device after state_dict export
    agent.actor.to(agent.device)

    trajectories = trajectories_from_dicts(traj_dicts)
    trajectory_groups = group_trajectories_by_instance_id(trajectories)

    summaries = write_diversity_trajectories_to_buffer(
        trajectory_groups=trajectory_groups,
        encoder=encoder,
        replay_buffer=replay_buffer,
        k_neighbors=k_neighbors,
    )
    return summaries


def train_preference_rl_full_3stage_parallel(
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
    max_workers=4,
    return_history=False,
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

    actor_class = SHyperActorFull
    actor_kwargs = {"hidden_dim": hidden_dim, "num_layers": num_layers}

    history = []

    # -------------------------------
    # Step 1: Diversity exploration
    # -------------------------------
    print("\n=== Step 1: Diversity exploration ===")
    for iteration in range(I1):
        seeds = list(range(next_seed, next_seed + batch_instance_size))
        next_seed += batch_instance_size

        summaries = collect_parallel_diversity_batch(
            seeds=seeds,
            agent=agent,
            encoder=encoder,
            replay_buffer=replay_buffer,
            num_trajectories_per_instance=num_trajectories_per_instance,
            k_neighbors=k_neighbors,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
            max_workers=max_workers,
            actor_class=actor_class,
            actor_kwargs=actor_kwargs,
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

        history.append({
            "stage": "step1_diversity",
            "iteration": iteration + 1,
            "buffer": int(len(replay_buffer)),
            "eval_avg": float(eval_avg),
            "q1_loss": None if avg_q1 is None else float(avg_q1),
            "q2_loss": None if avg_q2 is None else float(avg_q2),
            "actor_loss": None if avg_actor is None else float(avg_actor),
            "train_batch_seeds": [int(x) for x in seeds],
            "eval_makespans": [float(x) for x in eval_list],
        })

        print(
            f"[Step1] Iter {iteration + 1:03d} | "
            f"buffer={len(replay_buffer)} | "
            f"eval_avg={eval_avg:.4f}"
        )
        if avg_q1 is not None:
            print(
                f"  q1_loss={avg_q1:.4f} | q2_loss={avg_q2:.4f} | actor_loss={avg_actor:.4f}"
            )
        print("  train_batch_seeds:", seeds)
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
    history.append({
        "stage": "step2_reward_learning",
        "iteration": 0,
        "num_instances": int(summary["num_instances"]),
        "num_pairs": int(summary["num_pairs"]),
        "reward_loss": float(summary["reward_loss"]),
        "reward_acc": float(summary["reward_acc"]),
        "num_relabeled": int(summary["num_relabeled"]),
    })

    # -------------------------------
    # Step 3: Policy learning with reward_obj
    # -------------------------------
    print("\n=== Step 3: Policy learning with reward_obj ===")
    current_batch_seeds = []

    for iteration in range(I2):
        if (iteration % instance_refresh_interval == 0) or (len(current_batch_seeds) == 0):
            current_batch_seeds = list(range(next_seed, next_seed + batch_instance_size))
            next_seed += batch_instance_size

        summaries = collect_parallel_diversity_batch(
            seeds=current_batch_seeds,
            agent=agent,
            encoder=encoder,
            replay_buffer=replay_buffer,
            num_trajectories_per_instance=num_trajectories_per_instance,
            k_neighbors=k_neighbors,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
            max_workers=max_workers,
            actor_class=actor_class,
            actor_kwargs=actor_kwargs,
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
            summary = {
                "num_instances": 0,
                "num_pairs": 0,
                "reward_loss": 0.0,
                "reward_acc": 0.0,
                "created_new_model": False,
                "num_relabeled": 0,
            }

        if len(replay_buffer) >= batch_size:
            stats_list = []
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
            torch.save(agent.actor.state_dict(), "best_pref3stage_parallel_actor.pt")
            torch.save(agent.q1.state_dict(), "best_pref3stage_parallel_q1.pt")
            torch.save(agent.q2.state_dict(), "best_pref3stage_parallel_q2.pt")
            if reward_model is not None:
                torch.save(reward_model.state_dict(), "best_pref3stage_parallel_reward_model.pt")

        history.append({
            "stage": "step3_policy_learning",
            "iteration": iteration + 1,
            "buffer": int(len(replay_buffer)),
            "eval_avg": float(eval_avg),
            "best_eval_avg": float(best_eval_avg),
            "reward_pairs": int(summary["num_pairs"]),
            "reward_loss": float(summary["reward_loss"]),
            "reward_acc": float(summary["reward_acc"]),
            "num_relabeled": int(summary["num_relabeled"]),
            "q1_loss": None if avg_q1 is None else float(avg_q1),
            "q2_loss": None if avg_q2 is None else float(avg_q2),
            "actor_loss": None if avg_actor is None else float(avg_actor),
            "train_batch_seeds": [int(x) for x in current_batch_seeds],
            "eval_makespans": [float(x) for x in eval_list],
        })
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

    if return_history:
        meta = {
            "best_checkpoint_actor": "best_pref3stage_parallel_actor.pt",
            "best_checkpoint_q1": "best_pref3stage_parallel_q1.pt",
            "best_checkpoint_q2": "best_pref3stage_parallel_q2.pt",
            "best_checkpoint_reward_model": "best_pref3stage_parallel_reward_model.pt",
            "best_eval_avg": float(best_eval_avg),
        }
        return (agent, reward_model), history, meta

    return agent, reward_model


if __name__ == "__main__":
    train_preference_rl_full_3stage_parallel()