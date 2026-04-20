from env.instance_generator import generate_random_instance


def main():
    instance = generate_random_instance(seed=42)

    print("num_jobs:", instance.num_jobs)
    print("num_machines:", instance.num_machines)
    print("num_workers:", instance.num_workers)

    print("\nMachine automation:")
    print(instance.machine_automation)

    print("\nWorker physical condition:")
    print(instance.worker_physical_condition)

    print("\nJobs / operations:")
    for job in instance.jobs:
        for op in job:
            print(
                f"job={op.job_id}, op={op.op_id}, "
                f"machines={op.compatible_machines}, "
                f"workers={op.compatible_workers}, "
                f"base_times={op.base_processing_times}, "
                f"skills={op.skill_levels}, "
                f"difficulty={op.difficulty}"
            )


if __name__ == "__main__":
    main()