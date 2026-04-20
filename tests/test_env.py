from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv


def choose_shortest_action(env, actions):
    best_action = None
    best_time = float("inf")

    for (job_id, op_id, machine_id, worker_id) in actions:
        op = env.instance.jobs[job_id][op_id]
        base_time = op.base_processing_times[machine_id]
        skill = op.skill_levels[worker_id]
        fatigue = env.worker_fatigue[worker_id]

        est = base_time * (1.0 + __import__("math").log(1.0 + fatigue)) / skill
        if est < best_time:
            best_time = est
            best_action = (job_id, op_id, machine_id, worker_id)

    return best_action


def main():
    instance = create_demo_instance()
    env = FJSPWFEnv(instance)
    state = env.reset()

    print("Initial state:")
    print(state)

    step_id = 0
    while True:
        actions = env.get_valid_actions()
        print(f"\nDecision step {step_id}")
        print(f"Current time: {env.current_time}")
        print(f"Valid actions: {actions}")

        action = choose_shortest_action(env, actions)
        print(f"Choose action: {action}")

        state, reward, done, info = env.step(action)
        print("Next state:", state)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)

        step_id += 1
        if done:
            break

    print("\nFinal schedule:")
    for item in env.schedule:
        print(item)


if __name__ == "__main__":
    main()
    