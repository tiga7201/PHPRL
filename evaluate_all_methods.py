import os
import json
import torch
import time

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state

from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from rl.sac_agent import SACAgent

from rl.pdr_baselines import select_pdr_action

def get_pgnn_checkpoint_path():
    project_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(project_root, "checkpoints", "pgnn_phase1.pt")

def make_env(seed, num_jobs=3, num_machines=3, num_workers=3,
             min_ops_per_job=2, max_ops_per_job=4):
    instance = generate_random_instance(
        seed=seed,
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_workers=num_workers,
        min_ops_per_job=min_ops_per_job,
        max_ops_per_job=max_ops_per_job,
    )
    env = FJSPWFEnv(instance)
    env.load_pgnn_phase1(get_pgnn_checkpoint_path(), device="cpu")
    return env


def run_episode(env, agent):
    env.reset()
    done = False
    last_info = None

    start_time = time.perf_counter()

    while not done:
        graph_state = build_hypergraph_state(env)
        decision = agent.actor.select_greedy_action(graph_state)
        action = decision["action"]
        _, _, done, info = env.step(action)
        last_info = info

    solve_time = time.perf_counter() - start_time
    return float(last_info["makespan"]), float(solve_time)

def run_episode_pdr(env, rule: str):
    env.reset()
    done = False
    last_info = None

    start_time = time.perf_counter()

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            env._advance_to_next_event()
            valid_actions = env.get_valid_actions()

        action = select_pdr_action(env, rule)
        _, _, done, info = env.step(action)
        last_info = info

    solve_time = time.perf_counter() - start_time
    return float(last_info["makespan"]), float(solve_time)


def load_actor_checkpoint(actor, checkpoint_path, map_location="cpu"):
    """
    Support two checkpoint formats:
    1. pure actor state_dict
    2. full training checkpoint containing 'actor_state_dict'
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
        actor.load_state_dict(ckpt["actor_state_dict"])
    else:
        actor.load_state_dict(ckpt)


def evaluate_rl_method(
    actor_ckpt_path,
    seeds,
    hidden_dim=64,
    num_layers=2,
    num_jobs=3,
    num_machines=3,
    num_workers=3,
    min_ops_per_job=2,
    max_ops_per_job=4,
):
    actor = SHyperActorFull(hidden_dim=hidden_dim, num_layers=num_layers)
    q1 = SHyperQCriticFull(hidden_dim=hidden_dim, num_layers=num_layers)
    q2 = SHyperQCriticFull(hidden_dim=hidden_dim, num_layers=num_layers)

    agent = SACAgent(
        actor=actor,
        q1=q1,
        q2=q2,
        lr=1e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        device="cpu",
    )

    load_actor_checkpoint(actor, actor_ckpt_path, map_location="cpu")
    actor.eval()

    makespans = []
    solve_times = []

    for seed in seeds:
        env = make_env(
            seed=seed,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
        )
        makespan, solve_time = run_episode(env, agent)
        makespans.append(makespan)
        solve_times.append(solve_time)

    avg_makespan = sum(makespans) / len(makespans)
    worst_makespan = max(makespans)
    avg_solve_time = sum(solve_times) / len(solve_times)

    return {
        "avg_makespan": float(avg_makespan),
        "worst_makespan": float(worst_makespan),
        "avg_solve_time_seconds": float(avg_solve_time),
        "per_seed": [float(x) for x in makespans],
    }

def evaluate_pdr_method(
    rule,
    seeds,
    num_jobs=3,
    num_machines=3,
    num_workers=3,
    min_ops_per_job=2,
    max_ops_per_job=4,
):
    makespans = []
    solve_times = []

    for seed in seeds:
        env = make_env(
            seed=seed,
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_workers=num_workers,
            min_ops_per_job=min_ops_per_job,
            max_ops_per_job=max_ops_per_job,
        )
        makespan, solve_time = run_episode_pdr(env, rule)
        makespans.append(makespan)
        solve_times.append(solve_time)

    avg_makespan = sum(makespans) / len(makespans)
    worst_makespan = max(makespans)
    avg_solve_time = sum(solve_times) / len(solve_times)

    return {
        "avg_makespan": float(avg_makespan),
        "worst_makespan": float(worst_makespan),
        "avg_solve_time_seconds": float(avg_solve_time),
        "per_seed": [float(x) for x in makespans],
    }

def evaluate_ga_method(ga_json_path, expected_seeds):
    with open(ga_json_path, "r", encoding="utf-8") as f:
        ga_summary = json.load(f)

    ga_results = ga_summary["results"]
    ga_seed_to_result = {int(item["seed"]): float(item["best_makespan"]) for item in ga_results}

    makespans = []
    missing = []
    for seed in expected_seeds:
        if seed not in ga_seed_to_result:
            missing.append(seed)
        else:
            makespans.append(ga_seed_to_result[seed])

    if missing:
        raise ValueError(f"GA results missing seeds: {missing}")

    avg_makespan = sum(makespans) / len(makespans)
    worst_makespan = max(makespans)

    return {
        "avg_makespan": float(avg_makespan),
        "worst_makespan": float(worst_makespan),
        "per_seed": [float(x) for x in makespans],
        "source_json": ga_json_path,
    }


def main():
    # ===== unified evaluation config =====
    eval_config = {
        "seeds": list(range(500, 600)),
        "num_jobs": 40,
        "num_machines": 5,
        "num_workers": 3,
        "min_ops_per_job": 3,
        "max_ops_per_job": 7,
        "hidden_dim": 64,
        "num_layers": 2,
    }

    # ===== methods to evaluate =====
    methods = {
        # pure actor checkpoint
        "full_sac": {
            "type": "rl",
            "actor_ckpt": "best_fixedscale_full_sac_actor.pt",
        },

        # pure actor checkpoint
        "pref_3stage_parallel_best": {
            "type": "rl",
            "actor_ckpt": "best_pref3stage_parallel_actor.pt",
        },

        # archived full training checkpoints
        "pref_step3_iter_40": {
            "type": "rl",
            "actor_ckpt": "checkpoints/archive/ckpt_step3_iter_0040.pt",
        },
        "pref_step3_iter_0250": {
            "type": "rl",
            "actor_ckpt": "checkpoints/archive/ckpt_step3_iter_0250.pt",
        },

        "fifo": {
            "type": "pdr",
            "rule": "FIFO",
        },
        "spt": {
            "type": "pdr",
            "rule": "SPT",
        },
        "mwkr": {
            "type": "pdr",
            "rule": "MWKR",
        },

        # GA json result
        # "ga_baseline": {
        #     "type": "ga",
        #     "json_path": "eval_results/ga_baseline_summary.json",
        # },
    }

    results = {}

    for method_name, method_cfg in methods.items():
        print(f"\nEvaluating method: {method_name}")

        if method_cfg["type"] == "rl":
            ckpt_path = method_cfg["actor_ckpt"]
            if not os.path.exists(ckpt_path):
                print(f"Checkpoint not found: {ckpt_path}, skip.")
                continue

            result = evaluate_rl_method(
                actor_ckpt_path=ckpt_path,
                seeds=eval_config["seeds"],
                hidden_dim=eval_config["hidden_dim"],
                num_layers=eval_config["num_layers"],
                num_jobs=eval_config["num_jobs"],
                num_machines=eval_config["num_machines"],
                num_workers=eval_config["num_workers"],
                min_ops_per_job=eval_config["min_ops_per_job"],
                max_ops_per_job=eval_config["max_ops_per_job"],
            )

        elif method_cfg["type"] == "pdr":
            result = evaluate_pdr_method(
                rule=method_cfg["rule"],
                seeds=eval_config["seeds"],
                num_jobs=eval_config["num_jobs"],
                num_machines=eval_config["num_machines"],
                num_workers=eval_config["num_workers"],
                min_ops_per_job=eval_config["min_ops_per_job"],
                max_ops_per_job=eval_config["max_ops_per_job"],
            )

        elif method_cfg["type"] == "ga":
            json_path = method_cfg["json_path"]
            if not os.path.exists(json_path):
                print(f"GA result json not found: {json_path}, skip.")
                continue

            result = evaluate_ga_method(
                ga_json_path=json_path,
                expected_seeds=eval_config["seeds"],
            )

        else:
            raise ValueError(f"Unknown method type: {method_cfg['type']}")

        results[method_name] = result

        print(
            f"{method_name} | "
            f"avg_makespan={result['avg_makespan']:.4f} | "
            f"worst_makespan={result['worst_makespan']:.4f} | "
            f"avg_solve_time={result['avg_solve_time_seconds']:.6f}s"
        )
        print("per_seed:", [round(x, 4) for x in result["per_seed"]])

    ranked = sorted(
        results.items(),
        key=lambda kv: kv[1]["avg_makespan"]
    )

    print("\n=== Ranking by avg_makespan ===")
    for rank, (method_name, result) in enumerate(ranked, start=1):
        print(
            f"{rank}. {method_name} | "
            f"avg={result['avg_makespan']:.4f} | "
            f"worst={result['worst_makespan']:.4f}"
        )

    os.makedirs("eval_results", exist_ok=True)

    output = {
        "eval_config": eval_config,
        "results": results,
        "ranking": [
            {
                "rank": rank,
                "method": method_name,
                "avg_makespan": result["avg_makespan"],
                "worst_makespan": result["worst_makespan"],
            }
            for rank, (method_name, result) in enumerate(ranked, start=1)
        ],
    }

    with open("eval_results/evaluation_all_methods.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\nSaved to eval_results/evaluation_all_methods.json")


if __name__ == "__main__":
    main()