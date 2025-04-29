import numpy as np
import torch as T
from networks import ContinuousActorNetwork, ContinuousCriticNetwork


class Agent:
    def __init__(self, actor_dims, critic_dims,
                 n_actions, agent_idx, agent_name,
                 gamma=0.99, alpha=3e-4, T=2048,
                 gae_lambda=0.95, policy_clip=0.2,
                 batch_size=64, n_epochs=10,
                 n_procs=8, chkpt_dir=None,
                 scenario=None):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = 1e-3
        self.agent_idx = agent_idx
        self.agent_name = agent_name
        self.n_procs = n_procs

        self.actor = ContinuousActorNetwork(n_actions, actor_dims, alpha,
                                            chkpt_dir=chkpt_dir,
                                            scenario=scenario)
        self.critic = ContinuousCriticNetwork(critic_dims, alpha,
                                              chkpt_dir=chkpt_dir,
                                              scenario=scenario)
        self.n_actions = n_actions

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        with T.no_grad():
            state = T.tensor(observation, dtype=T.float,
                             device=self.actor.device)

            dist = self.actor(state)
            action = dist.sample()
            probs = dist.log_prob(action)
        return action.cpu().numpy(), probs.cpu().numpy()

    def calc_adv_and_returns(self, memories):
        states, new_states, r, dones = memories
        with T.no_grad():
            values = self.critic(states).squeeze()
            values_ = self.critic(new_states).squeeze()
            deltas = r[:, :, self.agent_idx] + self.gamma * values_ - values
            deltas = deltas.cpu().numpy()
            adv = [0]
            for step in reversed(range(deltas.shape[0])):
                advantage = deltas[step] +\
                    self.gamma*self.gae_lambda*adv[-1]*np.array(dones[step])
                adv.append(advantage)
            adv.reverse()
            adv = np.array(adv[:-1])
            adv = T.tensor(adv, device=self.critic.device).unsqueeze(2)
            returns = adv + values.unsqueeze(2)
            adv = (adv - adv.mean()) / (adv.std()+1e-4)
        return adv, returns

    def learn(self, memory):
        actor_states, states, actions, old_probs, rewards, actor_new_states, \
            states_, dones = memory.recall()
        device = self.critic.device
        state_arr = T.tensor(states, dtype=T.float, device=device)
        states__arr = T.tensor(states_, dtype=T.float, device=device)
        r = T.tensor(rewards, dtype=T.float, device=device)
        action_arr = T.tensor(actions[self.agent_name],
                              dtype=T.float, device=device)
        old_probs_arr = T.tensor(old_probs[self.agent_name], dtype=T.float,
                                 device=device)
        actor_states_arr = T.tensor(actor_states[self.agent_name],
                                    dtype=T.float, device=device)
        adv, returns = self.calc_adv_and_returns((state_arr, states__arr,
                                                 r, dones))
        for epoch in range(self.n_epochs):
            batches = memory.generate_batches()
            for batch in batches:
                old_probs = old_probs_arr[batch]
                actions = action_arr[batch]
                actor_states = actor_states_arr[batch]
                dist = self.actor(actor_states)
                new_probs = dist.log_prob(actions)
                prob_ratio = T.exp(new_probs.sum(2, keepdims=True) - old_probs.
                                   sum(2, keepdims=True))
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(
                        prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * \
                    adv[batch]
                entropy = dist.entropy().sum(2, keepdims=True)
                actor_loss = -T.min(weighted_probs,
                                    weighted_clipped_probs)
                actor_loss -= self.entropy_coefficient * entropy
                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()

                states = state_arr[batch]
                critic_value = self.critic(states).squeeze()
                critic_loss = \
                    (critic_value - returns[batch].squeeze()).pow(2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
