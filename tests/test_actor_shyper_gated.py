from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from models.actor_shyper_gated import SHyperActorGated


def main():
    instance = create_demo_instance()
    env = FJSPWFEnv(instance)
    env.reset()

    graph_state = build_hypergraph_state(env)

    actor = SHyperActorGated(hidden_dim=64)
    output = actor(graph_state)

    print("logits shape:", output["logits"].shape)
    print("probs shape:", output["probs"].shape)
    print("mask shape:", output["valid_action_mask"].shape)
    print("sum probs:", output["probs"].sum().item())

    print("\nValid action probabilities:")
    probs = output["probs"].tolist()
    mask = output["valid_action_mask"].tolist()
    for i, (p, m) in enumerate(zip(probs, mask)):
        if m > 0.5:
            print(i, p)


if __name__ == "__main__":
    main()