from rl.diversity_collector import collect_diversity_data_for_env
from rl.replay_buffer_pref import ReplayBufferPref
from rl.preference_pipeline import rebuild_trajectories_from_buffer
from rl.embedding_cache import cache_sa_embeddings_for_trajectory_groups
from rl.sac_agent import SACAgent

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv

from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from models.shypergnn_full import SHyperGNNFull
from models.reward_model import RewardModel


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

    encoder = SHyperGNNFull(hidden_dim=64, num_layers=2)
    replay_buffer = ReplayBufferPref(capacity=5000)

    for seed in [10, 11]:
        instance = generate_random_instance(
            seed=seed,
            num_jobs=3,
            num_machines=3,
            num_workers=3,
            min_ops_per_job=2,
            max_ops_per_job=4,
        )
        env = FJSPWFEnv(instance)

        collect_diversity_data_for_env(
            env=env,
            agent=agent,
            encoder=encoder,
            replay_buffer=replay_buffer,
            instance_id=f"inst_{seed}",
            num_trajectories=4,
            k_neighbors=3,
        )

    trajectory_groups = rebuild_trajectories_from_buffer(replay_buffer)
    reward_model = RewardModel(hidden_dim=64, num_layers=2)

    cache_sa_embeddings_for_trajectory_groups(trajectory_groups, reward_model.encoder)

    print("Num instance groups:", len(trajectory_groups))
    for i, trajs in enumerate(trajectory_groups):
        print(f"\nInstance group {i}")
        for j, traj in enumerate(trajs[:2]):
            print(
                f"traj {j} | steps={len(traj.graph_states)} | "
                f"cached_steps={0 if traj.sa_embeddings is None else len(traj.sa_embeddings)}"
            )

    # test fast scoring
    sample_traj = trajectory_groups[0][0]
    score = reward_model.score_trajectory(sample_traj)
    print("\nFast trajectory score OK:", float(score.item()))


if __name__ == "__main__":
    main()