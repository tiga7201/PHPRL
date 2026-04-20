from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv
from utils.graph_builder import build_hypergraph_state
from models.actor import BaselineActor


def main():
    instance = create_demo_instance()
    env = FJSPWFEnv(instance)
    actor = BaselineActor()

    state = env.reset()
    done = False
    step_id = 0
    trajectory = []

    while not done:
        graph_state = build_hypergraph_state(env)

        decision = actor.sample_action(graph_state)
        action = decision["action"]
        edge_idx = decision["edge_idx"]

        next_state, reward, done, info = env.step(action)

        trajectory.append({
            "step": step_id,
            "time": state["time"],
            "edge_idx": edge_idx,
            "action": action,
            "reward": reward,
            "done": done,
            "makespan": info["makespan"],
        })

        print(f"\nStep {step_id}")
        print("time:", state["time"])
        print("edge_idx:", edge_idx)
        print("action:", action)
        print("reward:", reward)
        print("done:", done)
        print("makespan:", info["makespan"])

        state = next_state
        step_id += 1

    print("\nFinal schedule:")
    for item in env.schedule:
        print(item)

    print("\nTrajectory:")
    for item in trajectory:
        print(item)


if __name__ == "__main__":
    main()