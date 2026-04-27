import os

from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pgnn_path = os.path.join(project_root, "checkpoints", "pgnn_phase1.pt")

    print("PGNN path:", pgnn_path)
    print("PGNN exists:", os.path.exists(pgnn_path))

    instance = create_demo_instance()
    env = FJSPWFEnv(instance)

    env.load_pgnn_phase1(pgnn_path, device="cpu")

    state = env.reset()

    print("Initial worker fatigue:", env.worker_fatigue)
    print("Initial worker workload:", env.worker_workload)
    print("Initial worker physical condition:", env.worker_physical_condition)
    print("Initial valid actions:", env.get_valid_actions())

    step_id = 0
    while not env.done and step_id < 5:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print("\nNo valid actions, advancing to next event...")
            old_time = env.current_time
            env._advance_to_next_event()
            print(f"Time advanced: {old_time:.4f} -> {env.current_time:.4f}")
            print("Worker fatigue after rest/event advance:", env.worker_fatigue)
            continue

        action = valid_actions[0]

        print(f"\n=== Step {step_id} ===")
        print("Current time:", env.current_time)
        print("Chosen action:", action)
        print("Worker fatigue before:", env.worker_fatigue)

        _, reward, done, info = env.step(action)

        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
        print("Worker fatigue after:", env.worker_fatigue)
        print("Worker workload after:", env.worker_workload)
        print("Current time after step:", env.current_time)

        step_id += 1

    print("\nFinal schedule:")
    for item in env.schedule:
        print(item)

    print("\nFinal worker fatigue:", env.worker_fatigue)
    print("Final worker workload:", env.worker_workload)


if __name__ == "__main__":
    main()