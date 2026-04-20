import json
import os
from datetime import datetime

from train_sac_shyper_full_fixedscale_batch import train_sac_shyper_full_fixedscale_batch
from train_preference_rl_full_3stage_parallel import train_preference_rl_full_3stage_parallel


def make_run_dir(experiment_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{timestamp}_{experiment_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_full_sac(config):
    agent, history, meta = train_sac_shyper_full_fixedscale_batch(
        num_iterations=config["num_iterations"],
        batch_instance_size=config["batch_instance_size"],
        episodes_per_instance_batch=config["episodes_per_instance_batch"],
        updates_per_episode_round=config["updates_per_episode_round"],
        instance_refresh_interval=config["instance_refresh_interval"],
        buffer_capacity=config["buffer_capacity"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        alpha=config["alpha"],
        num_jobs=config["num_jobs"],
        num_machines=config["num_machines"],
        num_workers=config["num_workers"],
        min_ops_per_job=config["min_ops_per_job"],
        max_ops_per_job=config["max_ops_per_job"],
        return_history=True,
    )
    return history, meta


def run_pref_3stage_parallel(config):
    (_, _), history, meta = train_preference_rl_full_3stage_parallel(
        I1=config["I1"],
        I2=config["I2"],
        batch_instance_size=config["batch_instance_size"],
        num_trajectories_per_instance=config["num_trajectories_per_instance"],
        k_neighbors=config["k_neighbors"],
        reward_model_epochs=config["reward_model_epochs"],
        updates_per_iter=config["updates_per_iter"],
        instance_refresh_interval=config["instance_refresh_interval"],
        buffer_capacity=config["buffer_capacity"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        alpha=config["alpha"],
        num_jobs=config["num_jobs"],
        num_machines=config["num_machines"],
        num_workers=config["num_workers"],
        min_ops_per_job=config["min_ops_per_job"],
        max_ops_per_job=config["max_ops_per_job"],
        max_workers=config["max_workers"],
        return_history=True,
    )
    return history, meta


def main():
    # ===== choose experiment here =====
    config = {
        "experiment_name": "pref_3stage_parallel",
        "method": "pref_3stage_parallel",

        # common
        "hidden_dim": 64,
        "num_layers": 2,
        "lr": 1e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.1,

        # fixed-scale instance setup
        "num_jobs": 10,
        "num_machines": 5,
        "num_workers": 3,
        "min_ops_per_job": 3,
        "max_ops_per_job": 8,

        # full_sac params
        "num_iterations": 20,
        "episodes_per_instance_batch": 5,
        "updates_per_episode_round": 2,

        # pref_3stage_parallel params
        "I1": 20,
        "I2": 1000,
        "num_trajectories_per_instance": 10,
        "k_neighbors": 5,
        "reward_model_epochs": 50,
        "updates_per_iter": 4,
        "max_workers": 24,

        # shared batch/buffer
        "batch_instance_size": 16,
        "instance_refresh_interval": 20,
        "buffer_capacity": 50000,
        "batch_size": 64,
    }

    run_dir = make_run_dir(config["experiment_name"])
    print("Run dir:", run_dir)

    save_json(config, os.path.join(run_dir, "config.json"))

    if config["method"] == "full_sac":
        history, meta = run_full_sac(config)
    elif config["method"] == "pref_3stage_parallel":
        history, meta = run_pref_3stage_parallel(config)
    else:
        raise ValueError(f"Unknown method: {config['method']}")

    save_json(history, os.path.join(run_dir, "history.json"))
    save_json(meta, os.path.join(run_dir, "meta.json"))

    summary = {
        "method": config["method"],
        "best_eval_avg": meta.get("best_eval_avg", None),
        "best_checkpoint_actor": meta.get("best_checkpoint_actor", None),
    }
    save_json(summary, os.path.join(run_dir, "summary.json"))

    print("Finished.")
    print("Summary:", summary)


if __name__ == "__main__":
    main()