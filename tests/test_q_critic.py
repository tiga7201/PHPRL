from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from models.q_critic import DiscreteQCritic


def main():
    instance = create_demo_instance()
    env = FJSPWFEnv(instance)
    env.reset()

    graph_state = build_hypergraph_state(env)

    critic = DiscreteQCritic()
    q_values = critic(graph_state)

    print("q_values shape:", q_values.shape)
    print(q_values)


if __name__ == "__main__":
    main()
