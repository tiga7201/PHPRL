import os
import torch

def save_light_checkpoint(
    path,
    agent,
    reward_model,
    reward_optimizer,
    stage,
    step1_iter,
    step3_iter,
    next_seed,
    current_batch_seeds,
    best_eval_avg,
    archive_dir=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "actor_state_dict": agent.actor.state_dict(),
        "q1_state_dict": agent.q1.state_dict(),
        "q2_state_dict": agent.q2.state_dict(),
        "target_q1_state_dict": agent.target_q1.state_dict(),
        "target_q2_state_dict": agent.target_q2.state_dict(),

        "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
        "q1_optimizer_state_dict": agent.q1_optimizer.state_dict(),
        "q2_optimizer_state_dict": agent.q2_optimizer.state_dict(),

        "reward_model_state_dict": None if reward_model is None else reward_model.state_dict(),
        "reward_optimizer_state_dict": None if reward_optimizer is None else reward_optimizer.state_dict(),

        "stage": stage,
        "step1_iter": step1_iter,
        "step3_iter": step3_iter,
        "next_seed": next_seed,
        "current_batch_seeds": current_batch_seeds,
        "best_eval_avg": best_eval_avg,
    }

    torch.save(checkpoint, path)

    # optionally archive every checkpoint with stage/iteration in filename
    if archive_dir is not None:
        os.makedirs(archive_dir, exist_ok=True)

        if stage == "step1":
            archive_name = f"ckpt_step1_iter_{step1_iter:04d}.pt"
        elif stage == "step3":
            archive_name = f"ckpt_step3_iter_{step3_iter:04d}.pt"
        else:
            archive_name = f"ckpt_{stage}.pt"

        archive_path = os.path.join(archive_dir, archive_name)
        torch.save(checkpoint, archive_path)


def load_light_checkpoint(
    path,
    agent,
    reward_model,
    reward_optimizer,
    device="cpu",
):
    checkpoint = torch.load(path, map_location=device)

    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.q1.load_state_dict(checkpoint["q1_state_dict"])
    agent.q2.load_state_dict(checkpoint["q2_state_dict"])
    agent.target_q1.load_state_dict(checkpoint["target_q1_state_dict"])
    agent.target_q2.load_state_dict(checkpoint["target_q2_state_dict"])

    agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    agent.q1_optimizer.load_state_dict(checkpoint["q1_optimizer_state_dict"])
    agent.q2_optimizer.load_state_dict(checkpoint["q2_optimizer_state_dict"])

    if reward_model is not None and checkpoint["reward_model_state_dict"] is not None:
        reward_model.load_state_dict(checkpoint["reward_model_state_dict"])

    if reward_optimizer is not None and checkpoint["reward_optimizer_state_dict"] is not None:
        reward_optimizer.load_state_dict(checkpoint["reward_optimizer_state_dict"])

    state = {
        "stage": checkpoint["stage"],
        "step1_iter": checkpoint["step1_iter"],
        "step3_iter": checkpoint["step3_iter"],
        "next_seed": checkpoint["next_seed"],
        "current_batch_seeds": checkpoint["current_batch_seeds"],
        "best_eval_avg": checkpoint["best_eval_avg"],
    }
    return state