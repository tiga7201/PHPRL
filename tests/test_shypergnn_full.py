from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from models.shypergnn_full import SHyperGNNFull


def main():
    instance = create_demo_instance()
    env = FJSPWFEnv(instance)
    env.reset()

    graph_state = build_hypergraph_state(env)

    encoder = SHyperGNNFull(hidden_dim=64, num_layers=2)
    out = encoder(graph_state)

    print("op_emb shape:", out["op_emb"].shape)
    print("machine_emb shape:", out["machine_emb"].shape)
    print("worker_emb shape:", out["worker_emb"].shape)
    print("edge_emb shape:", out["edge_emb"].shape)
    print("graph_emb shape:", out["graph_emb"].shape)


if __name__ == "__main__":
    main()