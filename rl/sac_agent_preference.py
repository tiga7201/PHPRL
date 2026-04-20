import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.actor import BaselineActor
from models.q_critic import DiscreteQCritic


class SACAgentPreference:
    def __init__(
        self,
        actor=None,
        q1=None,
        q2=None,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        device="cpu",
        pref_beta=0.1,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.pref_beta = pref_beta

        self.actor = (actor if actor is not None else BaselineActor()).to(self.device)
        self.q1 = (q1 if q1 is not None else DiscreteQCritic()).to(self.device)
        self.q2 = (q2 if q2 is not None else DiscreteQCritic()).to(self.device)

        self.target_q1 = copy.deepcopy(self.q1).to(self.device)
        self.target_q2 = copy.deepcopy(self.q2).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

    def select_action(self, graph_state, greedy=False):
        if greedy:
            decision = self.actor.select_greedy_action(graph_state)
        else:
            decision = self.actor.sample_action(graph_state)
        return decision["action"], decision["edge_idx"]

    def _compute_masked_policy(self, graph_state):
        actor_out = self.actor(graph_state)
        probs = actor_out["probs"]
        log_probs = torch.log(probs + 1e-8)
        mask = actor_out["valid_action_mask"]
        return probs, log_probs, mask

    def _compute_target_v(self, next_graph_state):
        with torch.no_grad():
            probs, log_probs, mask = self._compute_masked_policy(next_graph_state)

            q1_values = self.target_q1(next_graph_state)
            q2_values = self.target_q2(next_graph_state)
            q_min = torch.min(q1_values, q2_values)

            probs = probs * mask
            probs = probs / (probs.sum() + 1e-8)

            v_next = torch.sum(probs * (q_min - self.alpha * log_probs))
            return v_next

    def update(self, batch, reward_model=None):
        q1_losses = []
        q2_losses = []
        actor_losses = []

        for (state, action, edge_idx, reward, next_state, done) in batch:
            env_reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done_t = torch.tensor(float(done), dtype=torch.float32, device=self.device)

            # preference-based dense reward from reward model
            if reward_model is not None:
                with torch.no_grad():
                    pref_reward_t = reward_model.score_state_action(state, edge_idx)
                mixed_reward_t = (1.0 - self.pref_beta) * env_reward_t + self.pref_beta * pref_reward_t
            else:
                mixed_reward_t = env_reward_t

            # ----- critic update -----
            q1_values = self.q1(state)
            q2_values = self.q2(state)

            q1_sa = q1_values[edge_idx]
            q2_sa = q2_values[edge_idx]

            with torch.no_grad():
                if done:
                    target_q = mixed_reward_t
                else:
                    v_next = self._compute_target_v(next_state)
                    target_q = mixed_reward_t + self.gamma * (1.0 - done_t) * v_next

            q1_loss = F.mse_loss(q1_sa, target_q)
            q2_loss = F.mse_loss(q2_sa, target_q)

            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()

            q1_losses.append(q1_loss.item())
            q2_losses.append(q2_loss.item())

            # ----- actor update -----
            probs, log_probs, mask = self._compute_masked_policy(state)
            q1_curr = self.q1(state)
            q2_curr = self.q2(state)
            q_min = torch.min(q1_curr, q2_curr)

            probs = probs * mask
            probs = probs / (probs.sum() + 1e-8)

            actor_loss = torch.sum(probs * (self.alpha * log_probs - q_min))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())

            # ----- soft target update -----
            self.soft_update(self.target_q1, self.q1)
            self.soft_update(self.target_q2, self.q2)

        return {
            "q1_loss": sum(q1_losses) / len(q1_losses),
            "q2_loss": sum(q2_losses) / len(q2_losses),
            "actor_loss": sum(actor_losses) / len(actor_losses),
        }

    def soft_update(self, target_net, source_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                (1.0 - self.tau) * target_param.data + self.tau * source_param.data
            )