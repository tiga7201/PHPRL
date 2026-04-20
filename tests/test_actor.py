from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from models.actor import BaselineActor


def main():
    instance = create_demo_instance()
    env = FJSPWFEnv(instance)
    env.reset()

    graph_state = build_hypergraph_state(env)

    actor = BaselineActor()
    output = actor(graph_state)

    logits = output["logits"]
    probs = output["probs"]
    mask = output["valid_action_mask"]

    print("logits shape:", logits.shape)
    print("probs shape:", probs.shape)
    print("mask shape:", mask.shape)

    print("\nProbabilities:")
    print(probs)

    print("\nMasked probabilities (edge_idx, prob):")
    for i, (p, m) in enumerate(zip(probs.tolist(), mask.tolist())):
        if m > 0.5:
            print(i, p)

    print("\nSum of probs:", probs.sum().item())


if __name__ == "__main__":
    main()
