from dataclasses import dataclass
from typing import List

from rl.trajectory_collector import Trajectory


@dataclass
class PreferencePair:
    traj_better: Trajectory
    traj_worse: Trajectory
    label: int  # always 1 means traj_better is preferred over traj_worse


def build_preference_pairs(trajectories: List[Trajectory], min_gap: float = 0.0) -> List[PreferencePair]:
    """
    Build pairwise preferences from a list of trajectories of the same instance.

    Preference rule:
        lower makespan is better.
    min_gap:
        only construct a pair if worse.makespan - better.makespan >= min_gap
    """
    pairs = []

    n = len(trajectories)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            ti = trajectories[i]
            tj = trajectories[j]

            if ti.makespan + min_gap < tj.makespan:
                pairs.append(
                    PreferencePair(
                        traj_better=ti,
                        traj_worse=tj,
                        label=1,
                    )
                )

    return pairs


def build_preference_pairs_grouped(all_trajectory_groups, min_gap: float = 0.0):
    """
    all_trajectory_groups: List[List[Trajectory]]
        each inner list corresponds to one instance
    """
    all_pairs = []
    for trajs in all_trajectory_groups:
        all_pairs.extend(build_preference_pairs(trajs, min_gap=min_gap))
    return all_pairs