import os
import random
import numpy as np
import torch
import torch.optim as optim

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
from models.reward_model import RewardModel

from utils.checkpoint_utils_light import (
    save_light_checkpoint,
    load_light_checkpoint,
)


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


def collect_parallel_trajectory_groups(
    seeds,
    agent,
    num_trajectories_per_instance,
    num_jobs,
    num_machines,
    num_workers,
    min_ops_per_job,
    max_ops_per_job,
    max_workers,
    actor_class,
    actor_kwargs,
):
    original_device = agent.device
    agent.actor.to("cpu")

    traj_dicts = collect_trajectories_parallel(
        seeds=seeds,
        actor=agent.actor,
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

    agent.actor.to(original_device)

    trajectories = trajectories_from_dicts(traj_dicts)
    trajectory_groups = group_trajectories_by_instance_id(trajectories)
    return trajectory_groups

def write_one_instance_group_to_buffer(
    trajs,
    encoder,
    replay_buffer,
    k_neighbors=3,
):
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

def train_preference_rl_full_3stage_parallel(
    I1=5,
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
    resume_checkpoint_path=None,
    checkpoint_save_path="checkpoints/pref3stage_parallel_light.pt",
    checkpoint_save_interval=10,
    max_preference_pairs=1000,
    checkpoint_archive_dir="checkpoints/archive",
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
    reward_model = RewardModel(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    reward_optimizer = optim.Adam(reward_model.parameters(), lr=lr)

    replay_buffer = ReplayBufferPref(capacity=buffer_capacity)

    eval_seeds = [100, 101, 102, 103, 104]
    best_eval_avg = float("inf")

    next_seed = 0
    start_stage = "step1"
    start_step1_iter = 0
    start_step3_iter = 0
    current_batch_seeds = []

    history = []

    actor_class = SHyperActorFull
    actor_kwargs = {"hidden_dim": hidden_dim, "num_layers": num_layers}

    # 轻量恢复：只恢复模型/优化器/进度，不恢复 buffer
    if resume_checkpoint_path is not None and os.path.exists(resume_checkpoint_path):
        print(f"Resuming from light checkpoint: {resume_checkpoint_path}")
        state = load_light_checkpoint(
            path=resume_checkpoint_path,
            agent=agent,
            reward_model=reward_model,
            reward_optimizer=reward_optimizer,
            device=device,
        )
        start_stage = state["stage"]
        start_step1_iter = state["step1_iter"]
        start_step3_iter = state["step3_iter"]
        next_seed = state["next_seed"]
        current_batch_seeds = state["current_batch_seeds"]
        best_eval_avg = state["best_eval_avg"]

    # -------------------------------
    # Step 1: Diversity exploration
    # -------------------------------
    print("\n=== Step 1: Diversity exploration ===")
    if start_stage == "step1":
        step1_range = range(start_step1_iter, I1)
    else:
        step1_range = range(I1, I1)

    step1_refresh_interval = 30
    step1_current_batch_seeds = None

    for iteration in step1_range:
        if (iteration % step1_refresh_interval == 0) or (step1_current_batch_seeds is None):
            step1_current_batch_seeds = list(range(next_seed, next_seed + batch_instance_size))
            next_seed += batch_instance_size

        seeds = step1_current_batch_seeds

        trajectory_groups = collect_parallel_trajectory_groups(
            seeds=seeds,
            agent=agent,
            num_trajectories_per_instance=num_trajectories_per_instance,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
            max_workers=max_workers,
            actor_class=actor_class,
            actor_kwargs=actor_kwargs,
        )

        step_stats = []

        # 关键修改：每个实例完成 N_e 后，立刻做 N_g 次更新
        for trajs in trajectory_groups:
            write_one_instance_group_to_buffer(
                trajs=trajs,
                encoder=encoder,
                replay_buffer=replay_buffer,
                k_neighbors=k_neighbors,
            )

            if len(replay_buffer) >= batch_size:
                for _ in range(updates_per_iter):
                    batch = replay_buffer.sample(batch_size, reward_key="reward_div")
                    stat = agent.update(batch)
                    step_stats.append(stat)

        if len(step_stats) > 0:
            avg_q1 = sum(x["q1_loss"] for x in step_stats) / len(step_stats)
            avg_q2 = sum(x["q2_loss"] for x in step_stats) / len(step_stats)
            avg_actor = sum(x["actor_loss"] for x in step_stats) / len(step_stats)
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

        if ((iteration + 1) % checkpoint_save_interval == 0) or (iteration + 1 == I1):
            save_light_checkpoint(
                path=checkpoint_save_path,
                agent=agent,
                reward_model=reward_model,
                reward_optimizer=reward_optimizer,
                stage="step1" if (iteration + 1) < I1 else "step2",
                step1_iter=iteration + 1,
                step3_iter=0,
                next_seed=next_seed,
                current_batch_seeds=[],
                best_eval_avg=best_eval_avg,
                archive_dir=checkpoint_archive_dir,
            )

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
        max_preference_pairs=max_preference_pairs,
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

    save_light_checkpoint(
        path=checkpoint_save_path,
        agent=agent,
        reward_model=reward_model,
        reward_optimizer=reward_optimizer,
        stage="step3",
        step1_iter=I1,
        step3_iter=0,
        next_seed=next_seed,
        current_batch_seeds=current_batch_seeds,
        best_eval_avg=best_eval_avg,
        archive_dir=checkpoint_archive_dir,
    )

    # -------------------------------
    # Step 3: Policy learning with reward_obj
    # -------------------------------
    print("\n=== Step 3: Policy learning with reward_obj ===")
    if start_stage == "step3":
        step3_range = range(start_step3_iter, I2)
    else:
        step3_range = range(I2)

    for iteration in step3_range:
        if (iteration % instance_refresh_interval == 0) or (len(current_batch_seeds) == 0):
            current_batch_seeds = list(range(next_seed, next_seed + batch_instance_size))
            next_seed += batch_instance_size

        trajectory_groups = collect_parallel_trajectory_groups(
            seeds=current_batch_seeds,
            agent=agent,
            num_trajectories_per_instance=num_trajectories_per_instance,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
            max_workers=max_workers,
            actor_class=actor_class,
            actor_kwargs=actor_kwargs,
        )

        step_stats = []

        # 关键修改：每个实例完成 N_e 后，立刻做 N_g 次更新
        for trajs in trajectory_groups:
            write_one_instance_group_to_buffer(
                trajs=trajs,
                encoder=encoder,
                replay_buffer=replay_buffer,
                k_neighbors=k_neighbors,
            )

            if len(replay_buffer) >= batch_size:
                for _ in range(updates_per_iter):
                    batch = replay_buffer.sample(batch_size, reward_key="reward_obj")
                    stat = agent.update(batch)
                    step_stats.append(stat)

        # reward learning 仍然按周期触发，不跟着每个实例触发
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
                max_preference_pairs=max_preference_pairs,
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

        if len(step_stats) > 0:
            avg_q1 = sum(x["q1_loss"] for x in step_stats) / len(step_stats)
            avg_q2 = sum(x["q2_loss"] for x in step_stats) / len(step_stats)
            avg_actor = sum(x["actor_loss"] for x in step_stats) / len(step_stats)
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

        if ((iteration + 1) % checkpoint_save_interval == 0) or (iteration + 1 == I2):
            save_light_checkpoint(
                path=checkpoint_save_path,
                agent=agent,
                reward_model=reward_model,
                reward_optimizer=reward_optimizer,
                stage="step3",
                step1_iter=I1,
                step3_iter=iteration + 1,
                next_seed=next_seed,
                current_batch_seeds=current_batch_seeds,
                best_eval_avg=best_eval_avg,
                archive_dir=checkpoint_archive_dir,
            )

    if return_history:
        meta = {
            "best_checkpoint_actor": "best_pref3stage_parallel_actor.pt",
            "best_checkpoint_q1": "best_pref3stage_parallel_q1.pt",
            "best_checkpoint_q2": "best_pref3stage_parallel_q2.pt",
            "best_checkpoint_reward_model": "best_pref3stage_parallel_reward_model.pt",
            "best_eval_avg": float(best_eval_avg),
            "light_resume_checkpoint": checkpoint_save_path,
        }
        return (agent, reward_model), history, meta

    return agent, reward_model


if __name__ == "__main__":
    train_preference_rl_full_3stage_parallel()