from rl.diversity_collector import collect_diversity_data_for_env
from rl.replay_buffer_pref import ReplayBufferPref
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
    replay_buffer = ReplayBufferPref(capacity=5000)

    instance = generate_random_instance(
        seed=123,
        num_jobs=3,
        num_machines=3,
        num_workers=3,
        min_ops_per_job=2,
        max_ops_per_job=4,
    )
    env = FJSPWFEnv(instance)

    result = collect_diversity_data_for_env(
        env=env,
        agent=agent,
        encoder=encoder,
        replay_buffer=replay_buffer,
        instance_id="inst_123",
        num_trajectories=6,
        k_neighbors=3,
    )

    print("Summary:")
    print(result["summary"])

    print("\nReplay buffer size:", len(replay_buffer))

    first_items = replay_buffer.get_all()[:5]
    print("\nFirst few transitions:")
    for item in first_items:
        print({
            "instance_id": item["instance_id"],
            "traj_id": item["traj_id"],
            "step_id": item["step_id"],
            "reward_div": round(item["reward_div"], 4),
            "done": item["done"],
        })


if __name__ == "__main__":
    main()