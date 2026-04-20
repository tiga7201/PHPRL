import random
import torch

class ReplayBufferPref:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(
            self,
            state,
            action,
            edge_idx,
            next_state,
            done,
            reward_div=0.0,
            reward_obj=None,
            instance_id=None,
            traj_id=None,
            step_id=None,
            traj_makespan=None,
    ):
        transition = {
            "state": state,
            "action": action,
            "edge_idx": edge_idx,
            "next_state": next_state,
            "done": done,
            "reward_div": float(reward_div),
            "reward_obj": None if reward_obj is None else float(reward_obj),
            "instance_id": instance_id,
            "traj_id": traj_id,
            "step_id": step_id,
            "traj_makespan": None if traj_makespan is None else float(traj_makespan),
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, reward_key="reward_div"):
        samples = random.sample(self.buffer, batch_size)

        batch = []
        for item in samples:
            reward = item[reward_key]
            if reward is None:
                reward = 0.0

            batch.append((
                item["state"],
                item["action"],
                item["edge_idx"],
                reward,
                item["next_state"],
                item["done"],
            ))
        return batch

    def relabel_objective_rewards(self, reward_model):
        """
        Recompute reward_obj for all transitions using the learned reward model.
        """
        for item in self.buffer:
            state = item["state"]
            edge_idx = item["edge_idx"]
            with torch.no_grad():
                score = reward_model.score_state_action(state, edge_idx)
            item["reward_obj"] = float(score.item())

    def get_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)