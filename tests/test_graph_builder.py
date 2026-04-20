from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state


def main():
    instance = create_demo_instance()
    env = FJSPWFEnv(instance)
    env.reset()

    graph_state = build_hypergraph_state(env)

    print("op_features shape:", graph_state["op_features"].shape)
    print("machine_features shape:", graph_state["machine_features"].shape)
    print("worker_features shape:", graph_state["worker_features"].shape)
    print("edge_features shape:", graph_state["edge_features"].shape)
    print("edge_links shape:", graph_state["edge_links"].shape)
    print("valid_action_mask shape:", graph_state["valid_action_mask"].shape)

    print("\nFirst few edge links:")
    print(graph_state["edge_links"][:10])

    print("\nValid action mask:")
    print(graph_state["valid_action_mask"])

    print("\nAction to edge:")
    for k, v in list(graph_state["action_to_edge"].items())[:10]:
        print(k, "->", v)


if __name__ == "__main__":
    main()
    