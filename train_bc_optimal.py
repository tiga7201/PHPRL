import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from models.actor_shyper import SHyperActor
from exact.optimal_dataset import build_optimal_demo_dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_greedy(env, actor):
    state = env.reset()
    done = False

    while not done:
        graph_state = build_hypergraph_state(env)
        decision = actor.select_greedy_action(graph_state)
        action = decision["action"]
        state, reward, done, info = env.step(action)

    return info["makespan"]


def train_bc(num_epochs=500, lr=1e-3, hidden_dim=64):
    set_seed(42)

    dataset, result = build_optimal_demo_dataset()

    actor = SHyperActor(hidden_dim=hidden_dim)
    optimizer = optim.Adam(actor.parameters(), lr=lr)

    best_greedy = float("inf")

    for epoch in range(num_epochs):
        total_loss = 0.0

        for sample in dataset:
            graph_state = sample["graph_state"]
            target_edge_idx = sample["optimal_edge_idx"]

            out = actor(graph_state)
            logits = out["logits"].unsqueeze(0)  # [1, num_edges]
            target = torch.tensor([target_edge_idx], dtype=torch.long)

            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            env = FJSPWFEnv(create_demo_instance())
            greedy_makespan = evaluate_greedy(env, actor)
            best_greedy = min(best_greedy, greedy_makespan)

            print(
                f"Epoch {epoch + 1:03d} | "
                f"loss={total_loss:.4f} | "
                f"greedy={greedy_makespan:.4f} | "
                f"best_greedy={best_greedy:.4f}"
            )

    torch.save(actor.state_dict(), "bc_optimal_actor.pt")
    return actor


if __name__ == "__main__":
    train_bc()