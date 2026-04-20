from rl.preference_batch_collector import (
    build_training_batch_envs,
    collect_preference_data_for_batch,
)
from rl.sac_agent import SACAgent
from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull


def main():
    actor = SHyperActorFull(hidden_dim=64, num_layers=2)
    q1 = SHyperQCriticFull(hidden_dim=64, num_layers=2)
    q2 = SHyperQCriticFull(hidden_dim=64, num_layers=2)

    agent = SACAgent(
        actor=actor,
        q1=q1,
        q2=q2,
        lr=1e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        device="cpu",
    )

    envs, instance_ids = build_training_batch_envs(
        start_seed=0,
        batch_instance_size=4,
        num_jobs=3,
        num_machines=3,
        num_workers=3,
        min_ops_per_job=2,
        max_ops_per_job=4,
    )

    data = collect_preference_data_for_batch(
        envs=envs,
        instance_ids=instance_ids,
        agent=agent,
        num_trajectories_per_instance=6,
        min_gap=0.0,
    )

    print("Total instances:", len(data["trajectory_groups"]))
    print("Total preference pairs:", len(data["all_pairs"]))

    print("\nPer-instance summary:")
    for item in data["summary"]:
        print(item)


if __name__ == "__main__":
    main()