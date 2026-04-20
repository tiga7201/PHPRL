from utils.graph_builder import build_hypergraph_state

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Trajectory:
    instance_id: str
    graph_states: List[Dict[str, Any]]
    edge_indices: List[int]
    actions: List[tuple]
    rewards: List[float]
    makespan: float

    sa_embeddings: Optional[List[Any]] = field(default=None)
    transition_refs: Optional[List[Any]] = field(default=None)


def collect_trajectory(env, agent, instance_id: str, greedy: bool = False) -> Trajectory:
    """
    Collect one trajectory from a given env using the current policy.
    """
    env.reset()
    done = False

    graph_states = []
    edge_indices = []
    actions = []
    rewards = []

    final_makespan = None

    while not done:
        graph_state = build_hypergraph_state(env)

        if greedy:
            decision = agent.actor.select_greedy_action(graph_state)
        else:
            decision = agent.actor.sample_action(graph_state)

        action = decision["action"]
        edge_idx = decision["edge_idx"]

        _, reward, done, info = env.step(action)

        graph_states.append(graph_state)
        edge_indices.append(edge_idx)
        actions.append(action)
        rewards.append(float(reward))

        final_makespan = float(info["makespan"])

    return Trajectory(
        instance_id=instance_id,
        graph_states=graph_states,
        edge_indices=edge_indices,
        actions=actions,
        rewards=rewards,
        makespan=final_makespan,
    )


def collect_multiple_trajectories_for_env(env, agent, instance_id: str, num_trajectories: int):
    """
    Repeatedly solve the same instance to obtain multiple trajectories.
    """
    trajectories = []
    for _ in range(num_trajectories):
        traj = collect_trajectory(env, agent, instance_id=instance_id, greedy=False)
        trajectories.append(traj)
    return trajectories