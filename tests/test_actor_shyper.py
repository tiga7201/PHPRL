from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from models.actor_shyper import SHyperActor


def main():
    instance = create_demo_instance()
    env = FJSPWFEnv(instance)
    env.reset()

    graph_state = build_hypergraph_state(env)

    actor = SHyperActor(hidden_dim=64)
    output = actor(graph_state)

    logits = output["logits"]
    probs = output["probs"]
    mask = output["valid_action_mask"]

    print("logits shape:", logits.shape)
    print("probs shape:", probs.shape)
    print("mask shape:", mask.shape)
    print("sum probs:", probs.sum().item())

    print("\nValid action probabilities:")
    for i, (p, m) in enumerate(zip(probs.tolist(), mask.tolist())):
        if m > 0.5:
            print(i, p)


if __name__ == "__main__":
    main()