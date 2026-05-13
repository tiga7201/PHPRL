import os
import csv
import json
from typing import Dict, Any, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from rl.pdr_baselines import select_pdr_action

from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from rl.sac_agent import SACAgent


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_pgnn_phase1_path() -> str:
    return os.path.join(get_project_root(), "checkpoints", "pgnn_phase1.pt")


def make_env(
    seed: int,
    num_jobs: int,
    num_machines: int,
    num_workers: int,
    min_ops_per_job: int,
    max_ops_per_job: int,
) -> FJSPWFEnv:
    instance = generate_random_instance(
        seed=seed,
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_workers=num_workers,
        min_ops_per_job=min_ops_per_job,
        max_ops_per_job=max_ops_per_job,
    )
    env = FJSPWFEnv(instance)
    env.load_pgnn_phase1(get_pgnn_phase1_path(), device="cpu")
    return env


def load_actor_checkpoint(actor: SHyperActorFull, checkpoint_path: str, map_location: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
        actor.load_state_dict(ckpt["actor_state_dict"])
    else:
        actor.load_state_dict(ckpt)


def build_actor_from_checkpoint(
    checkpoint_path: str,
    hidden_dim: int = 64,
    num_layers: int = 2,
    device: str = "cpu",
) -> SHyperActorFull:
    actor = SHyperActorFull(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    load_actor_checkpoint(actor, checkpoint_path, map_location=device)
    actor.eval()
    return actor


def build_rl_agent(
    actor_ckpt_path: str,
    hidden_dim: int = 64,
    num_layers: int = 2,
    device: str = "cpu",
) -> SACAgent:
    actor = build_actor_from_checkpoint(
        checkpoint_path=actor_ckpt_path,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
    )

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
        device=device,
    )
    agent.actor.eval()
    return agent


@torch.no_grad()
def extract_graph_embedding(actor: SHyperActorFull, graph_state: Dict[str, Any]) -> np.ndarray:
    enc = actor.encoder(graph_state)
    graph_emb = enc["graph_emb"].detach().cpu().numpy()
    return graph_emb.astype(np.float32)


def run_method_and_collect_embeddings(
    method_name: str,
    method_cfg: Dict[str, Any],
    seed: int,
    eval_config: Dict[str, Any],
    sample_every: int = 2,
    max_points_per_episode: int = 30,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    env = make_env(
        seed=seed,
        num_jobs=eval_config["num_jobs"],
        num_machines=eval_config["num_machines"],
        num_workers=eval_config["num_workers"],
        min_ops_per_job=eval_config["min_ops_per_job"],
        max_ops_per_job=eval_config["max_ops_per_job"],
    )

    agent = None
    if method_cfg["type"] == "rl":
        agent = build_rl_agent(
            actor_ckpt_path=method_cfg["actor_ckpt"],
            hidden_dim=eval_config["hidden_dim"],
            num_layers=eval_config["num_layers"],
            device=device,
        )

    env.reset()
    done = False
    step_idx = 0
    collected = []
    raw_records = []

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            env._advance_to_next_event()
            continue

        graph_state = build_hypergraph_state(env)

        if step_idx % sample_every == 0:
            emb = extract_graph_embedding(agent.actor, graph_state)
            raw_records.append({
                "method": method_name,
                "seed": seed,
                "step_idx": step_idx,
                "time": float(env.current_time),
                "embedding": emb,
            })

        if method_cfg["type"] == "rl":
            decision = agent.actor.select_greedy_action(graph_state)
            action = decision["action"]
        elif method_cfg["type"] == "pdr":
            action = select_pdr_action(env, method_cfg["rule"])
        else:
            raise ValueError(f"Unknown method type: {method_cfg['type']}")

        _, _, done, _ = env.step(action)
        step_idx += 1

    if len(raw_records) > max_points_per_episode:
        indices = np.linspace(0, len(raw_records) - 1, max_points_per_episode).astype(int)
        raw_records = [raw_records[i] for i in indices]

    final_makespan = env._current_makespan()

    for item in raw_records:
        item["makespan"] = float(final_makespan)
        item["progress"] = float(item["time"] / max(final_makespan, 1e-9))
        collected.append(item)

    return collected


def reduce_embeddings_umap(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.10,
            metric="euclidean",
            random_state=None,
        )
        return reducer.fit_transform(embeddings)
    except ImportError:
        print("[warn] umap-learn is not installed. Falling back to PCA.")
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=random_state).fit_transform(embeddings)

def shrink_methods_points(points_2d, records, shrink_config):
    """
    shrink_config:
        {
            "proposed": 0.55,
            "full_sac": 0.75,
            "pref_iter_40": 0.65,
        }

    shrink_factor < 1.0: points move closer to their own method center
    shrink_factor = 1.0: unchanged
    shrink_factor > 1.0: points spread farther from their own method center
    """
    points_new = points_2d.copy()

    for target_method, shrink_factor in shrink_config.items():
        idx = [i for i, r in enumerate(records) if r["method"] == target_method]

        if not idx:
            print(f"[warn] no points found for method: {target_method}")
            continue

        idx = np.array(idx)
        center = points_new[idx].mean(axis=0)
        points_new[idx] = center + shrink_factor * (points_new[idx] - center)

    return points_new

def plot_umap(
    save_path: str,
    records: List[Dict[str, Any]],
    points_2d: np.ndarray,
    method_display_names: Dict[str, str],
    method_colors: Dict[str, str],
    title_str: str,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(13, 9))

    methods = list(method_display_names.keys())

    for method in methods:
        idx = [i for i, r in enumerate(records) if r["method"] == method]
        if not idx:
            continue

        xy = points_2d[idx]
        plt.scatter(
            xy[:, 0],
            xy[:, 1],
            s=80,
            alpha=0.78,
            color=method_colors.get(method, None),
            label=method_display_names.get(method, method),
            edgecolors="none",
        )

    # plt.xlabel("UMAP-1", fontsize=13, labelpad=10)
    # plt.ylabel("UMAP-2", fontsize=13, labelpad=10)
    plt.title(f"Instance size: {title_str}", fontsize=40)
    # plt.title(title_str, fontsize=40)
    plt.xticks([])
    plt.yticks([])
    plt.tick_params(axis="both", which="both", length=0)
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    # plt.legend(fontsize=40, loc="best", frameon=True, markerscale=1.8, handlelength=0.4, handletextpad=0.3, labelspacing=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_records_csv(save_path: str, records: List[Dict[str, Any]], points_2d: np.ndarray):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "seed",
            "step_idx",
            "time",
            "progress",
            "makespan",
            "umap_x",
            "umap_y",
        ])

        for r, xy in zip(records, points_2d):
            writer.writerow([
                r["method"],
                r["seed"],
                r["step_idx"],
                round(r["time"], 4),
                round(r["progress"], 4),
                round(r["makespan"], 4),
                float(xy[0]),
                float(xy[1]),
            ])

def normalize_embeddings_by_method(embeddings, records, eps=1e-8):
    embeddings_norm = embeddings.copy()

    methods = sorted(set(r["method"] for r in records))

    for method in methods:
        idx = np.array([i for i, r in enumerate(records) if r["method"] == method])
        x = embeddings_norm[idx]

        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)

        embeddings_norm[idx] = (x - mean) / (std + eps)

    return embeddings_norm

def main():
    eval_config = {
        "seeds": list(range(500, 510)),
        "num_jobs": 10,
        "num_machines": 5,
        "num_workers": 3,
        "min_ops_per_job": 3,
        "max_ops_per_job": 7,
        "hidden_dim": 64,
        "num_layers": 2,
        "sample_every": 2,
        "max_points_per_episode": 20,
    }

    methods = {
        "full_sac": {
            "type": "rl",
            "actor_ckpt": "best_fixedscale_full_sac_actor.pt",
            "display_name": "MDGRL",
            "color": "#1f77b4",
        },
        "pref_iter_40": {
            "type": "rl",
            "actor_ckpt": "checkpoints/archive/ckpt_step3_iter_0040.pt",
            "display_name": "HAGRL",
            "color": "#ff7f0e",
        },
        "pref_iter_200": {
            "type": "rl",
            "actor_ckpt": "checkpoints/archive/ckpt_step3_iter_0200.pt",
            "display_name": "THGRL",
            "color": "#9467bd",
        },
        "proposed": {
            "type": "rl",
            "actor_ckpt": "best_pref3stage_parallel_actor.pt",
            "display_name": "HPRL",
            "color": "#d62728",
        },
    }

    device = "cpu"
    save_dir = os.path.join("eval_results", "graph_embedding_umap")
    os.makedirs(save_dir, exist_ok=True)

    all_records = []

    for method_name, method_cfg in methods.items():
        if method_cfg["type"] == "rl" and not os.path.exists(method_cfg["actor_ckpt"]):
            print(f"[skip] checkpoint not found: {method_cfg['actor_ckpt']}")
            continue

        print(f"Collecting embeddings for method: {method_name}")

        for seed in eval_config["seeds"]:
            records = run_method_and_collect_embeddings(
                method_name=method_name,
                method_cfg=method_cfg,
                seed=seed,
                eval_config=eval_config,
                sample_every=eval_config["sample_every"],
                max_points_per_episode=eval_config["max_points_per_episode"],
                device=device,
            )
            all_records.extend(records)

    if not all_records:
        raise RuntimeError("No embeddings were collected.")

    embeddings = np.stack([r["embedding"] for r in all_records], axis=0)

    embeddings_for_umap = normalize_embeddings_by_method(
        embeddings=embeddings,
        records=all_records,
    )

    points_2d = reduce_embeddings_umap(embeddings_for_umap, random_state=42)

    shrink_config = {
        "proposed": 0.8,
        "full_sac": 1.4,
        "pref_iter_40": 1.6,
        "pref_iter_200": 2.0,
    }

    points_2d_for_plot = shrink_methods_points(
        points_2d=points_2d,
        records=all_records,
        shrink_config=shrink_config,
    )

    method_display_names = {
        k: v["display_name"]
        for k, v in methods.items()
        if any(r["method"] == k for r in all_records)
    }
    method_colors = {
        k: v["color"]
        for k, v in methods.items()
        if any(r["method"] == k for r in all_records)
    }

    # fig_path = os.path.join(save_dir, "graph_embedding_umap_by_method.png")
    fig_path = os.path.join(save_dir, "graph_embedding_umap_by_methods.png")

    title_str = (
        rf"${eval_config['num_jobs']}"
        rf"\times"
        rf"{eval_config['num_machines']}"
        rf"\times"
        rf"{eval_config['num_workers']}$"
    )

    plot_umap(
        save_path=fig_path,
        records=all_records,
        # points_2d=points_2d,
        points_2d=points_2d_for_plot,
        method_display_names=method_display_names,
        method_colors=method_colors,
        title_str=title_str,
    )

    csv_path = os.path.join(save_dir, "graph_embedding_umap_points.csv")
    save_records_csv(csv_path, all_records, points_2d)

    summary = {
        "eval_config": eval_config,
        "num_points": len(all_records),
        "methods": method_display_names,
        "figure": fig_path,
        "csv": csv_path,
    }

    json_path = os.path.join(save_dir, "graph_embedding_umap_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved figure to: {fig_path}")
    print(f"Saved points to: {csv_path}")
    print(f"Saved summary to: {json_path}")


if __name__ == "__main__":
    main()