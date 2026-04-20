from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from rl.replay_buffer import ReplayBuffer
from rl.sac_agent import SACAgent


def collect_one_episode(env, agent, buffer):
    state = env.reset()
    done = False

    while not done:
        graph_state = build_hypergraph_state(env)
        action, edge_idx = agent.select_action(graph_state, greedy=False)

        next_state, reward, done, info = env.step(action)
        next_graph_state = build_hypergraph_state(env)

        buffer.add(
            state=graph_state,
            action=action,
            edge_idx=edge_idx,
            reward=reward,
            next_state=next_graph_state,
            done=done,
        )

        state = next_state


def main():
    env = FJSPWFEnv(create_demo_instance())
    agent = SACAgent()
    buffer = ReplayBuffer(capacity=1000)

    for _ in range(10):
        collect_one_episode(env, agent, buffer)

    print("buffer size:", len(buffer))

    batch = buffer.sample(batch_size=4)
    stats = agent.update(batch)

    print("update stats:", stats)


if __name__ == "__main__":
    main()