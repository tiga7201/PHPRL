from env.instance_generator import generate_random_instance, save_instance_dataset


def main():
    seeds = [100, 101, 102, 103, 104]
    instances = []

    for seed in seeds:
        inst = generate_random_instance(
            seed=seed,
            num_jobs=10,
            num_machines=5,
            num_workers=3,
            min_ops_per_job=3,
            max_ops_per_job=7,
        )
        instances.append(inst)

    save_instance_dataset(instances, "benchmark_exact_3x3x3.json")
    print("Saved benchmark_exact_3x3x3.json with", len(instances), "instances")


if __name__ == "__main__":
    main()