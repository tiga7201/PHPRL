from copy import deepcopy

from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv


class BruteForceSolver:
    def __init__(self, env: FJSPWFEnv):
        self.initial_env = env
        self.best_makespan = float("inf")
        self.best_schedule = None
        self.best_action_seq = None
        self.num_nodes_searched = 0

    def solve(self):
        env = deepcopy(self.initial_env)
        env.reset()
        self._dfs(env, action_seq=[])
        return {
            "best_makespan": self.best_makespan,
            "best_schedule": self.best_schedule,
            "best_action_seq": self.best_action_seq,
            "num_nodes_searched": self.num_nodes_searched,
        }

    def _dfs(self, env: FJSPWFEnv, action_seq):
        self.num_nodes_searched += 1

        if env.done:
            makespan = max((op.end for op in env.schedule), default=0.0)
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_schedule = deepcopy(env.schedule)
                self.best_action_seq = action_seq.copy()
            return

        # simple branch-and-bound
        current_partial_makespan = max((op.end for op in env.schedule), default=0.0)
        if current_partial_makespan >= self.best_makespan:
            return

        valid_actions = env.get_valid_actions()

        # optional heuristic ordering: try shorter action first
        def action_key(action):
            job_id, op_id, machine_id, worker_id = action
            op = env.instance.jobs[job_id][op_id]
            base_time = op.base_processing_times[machine_id]
            skill = op.skill_levels[worker_id]
            fatigue = env.worker_fatigue[worker_id]

            import math
            est = base_time * (1.0 + math.log(1.0 + fatigue)) / skill
            return est

        valid_actions = sorted(valid_actions, key=action_key)

        for action in valid_actions:
            next_env = deepcopy(env)
            _, _, _, info = next_env.step(action)
            self._dfs(next_env, action_seq + [action])


def main():
    instance = create_demo_instance()
    env = FJSPWFEnv(instance)

    solver = BruteForceSolver(env)
    result = solver.solve()

    print("Best makespan:", result["best_makespan"])
    print("Nodes searched:", result["num_nodes_searched"])

    print("\nBest action sequence:")
    for a in result["best_action_seq"]:
        print(a)

    print("\nBest schedule:")
    for item in result["best_schedule"]:
        print(item)


if __name__ == "__main__":
    main()