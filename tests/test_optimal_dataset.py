from exact.optimal_dataset import build_optimal_demo_dataset


def main():
    dataset, result = build_optimal_demo_dataset()

    print("Optimal makespan:", result["best_makespan"])
    print("Num samples:", len(dataset))

    for i, sample in enumerate(dataset):
        print(f"\nStep {i}")
        print("Optimal action:", sample["optimal_action"])
        print("Optimal edge idx:", sample["optimal_edge_idx"])
        print("Mask sum:", sample["graph_state"]["valid_action_mask"].sum())


if __name__ == "__main__":
    main()