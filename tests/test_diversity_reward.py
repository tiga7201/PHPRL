from rl.trajectory_collector import collect_multiple_trajectories_for_env
from rl.diversity_reward import compute_diversity_rewards_for_trajectories
from rl.sac_agent import SACAgent

from env.instance_generator import generate_random_instance
from env.fjspwf_env import FJSPWFEnv

from models.actor_shyper_full import SHyperActorFull
from models.q_critic_shyper_full import SHyperQCriticFull
from models.shypergnn_full import SHyperGNNFull


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

    instance = generate_random_instance(
        seed=123,
        num_jobs=3,
        num_machines=3,
        num_workers=3,
        min_ops_per_job=2,
        max_ops_per_job=4,
    )
    env = FJSPWFEnv(instance)

    trajectories = collect_multiple_trajectories_for_env(
        env=env,
        agent=agent,
        instance_id="inst_123",
        num_trajectories=6,
    )

    rewards_per_traj = compute_diversity_rewards_for_trajectories(
        trajectories=trajectories,
        encoder=encoder,
        k_neighbors=3,
    )

    print("Num trajectories:", len(trajectories))
    for i, (traj, rewards) in enumerate(zip(trajectories, rewards_per_traj)):
        print(f"\nTrajectory {i}")
        print("makespan:", round(traj.makespan, 4))
        print("num_steps:", len(traj.graph_states))
        print("div_rewards:", [round(r, 4) for r in rewards])


if __name__ == "__main__":
    main()