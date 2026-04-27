import os

from env.instance_generator import create_demo_instance
from env.fjspwf_env import FJSPWFEnv


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pgnn_path = os.path.join(project_root, "checkpoints", "pgnn_phase1.pt")

    instance = create_demo_instance()
    env = FJSPWFEnv(instance)
    env.load_pgnn_phase1(pgnn_path, device="cpu")
    env.reset()

    # 手动设置一个恢复场景
    env.current_time = 0.0
    env.worker_fatigue[0] = 0.6
    env.worker_fatigue[1] = 0.0

    # worker 0 空闲，worker 1 正在忙到 5.0
    env.worker_available[0] = 0.0
    env.worker_available[1] = 5.0

    # 为了让 _advance_to_next_event() 有 future event，
    # 需要环境里有一个“运行中的任务结束时刻”
    env.schedule = []
    from env.fjspwf_env import ScheduledOp
    env.schedule.append(
        ScheduledOp(
            job_id=999,
            op_id=0,
            machine_id=0,
            worker_id=1,
            start=0.0,
            end=5.0,
            proc_time=5.0,
        )
    )

    print("Before recovery:", env.worker_fatigue)
    print("Current time:", env.current_time)
    print("Worker available:", env.worker_available)

    env._advance_to_next_event()

    print("After recovery:", env.worker_fatigue)
    print("New current time:", env.current_time)


if __name__ == "__main__":
    main()