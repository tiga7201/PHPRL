from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from models.actor_shyper_full import SHyperActorFull


def main():
    instance = create_demo_instance()
    env = FJSPWFEnv(instance)
    env.reset()

    graph_state = build_hypergraph_state(env)

    actor = SHyperActorFull(hidden_dim=64, num_layers=2)
    output = actor(graph_state)

    print("logits shape:", output["logits"].shape)
    print("probs shape:", output["probs"].shape)
    print("mask shape:", output["valid_action_mask"].shape)
    print("sum probs:", output["probs"].sum().item())


if __name__ == "__main__":
    main()