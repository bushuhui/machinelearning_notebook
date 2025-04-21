import numpy as np


class PPOMemory:
    def __init__(self, batch_size, T, n_agents, agents, n_procs,
                 critic_dims, actor_dims, n_actions):

        self.states = np.zeros((T, n_procs, critic_dims), dtype=np.float32)
        self.rewards = np.zeros((T, n_procs, n_agents), dtype=np.float32)
        self.dones = np.zeros((T, n_procs), dtype=np.float32)
        self.new_states = np.zeros((T, n_procs, critic_dims), dtype=np.float32)

        self.actor_states = {a: np.zeros((T, n_procs, actor_dims[a]))
                             for a in agents}
        self.actor_new_states = {a: np.zeros((T, n_procs, actor_dims[a]))
                                 for a in agents}
        self.actions = {a: np.zeros((T, n_procs, n_actions[a]))
                        for a in agents}
        self.probs = {a: np.zeros((T, n_procs, n_actions[a]))
                      for a in agents}

        self.mem_cntr = 0
        self.n_states = T
        self.n_procs = n_procs
        self.critic_dims = critic_dims
        self.actor_dims = actor_dims
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.agents = agents
        self.batch_size = batch_size

    def recall(self):
        return self.actor_states, \
            self.states, \
            self.actions, \
            self.probs, \
            self.rewards, \
            self.actor_new_states, \
            self.new_states, \
            self.dones

    def generate_batches(self):
        # batch_start = np.arange(0, n_states, self.batch_size)
        n_batches = int(self.n_states // self.batch_size)
        indices = np.arange(self.n_states, dtype=np.int64)
        np.random.shuffle(indices)
        # batches = [indices[i:i+self.batch_size] for i in batch_start]
        batches = [indices[i*self.batch_size:(i+1)*self.batch_size]
                   for i in range(n_batches)]
        return batches

    def store_memory(self, raw_obs, state, action, probs, reward,
                     raw_obs_, state_, done):
        index = self.mem_cntr % self.n_states
        self.states[index] = state
        self.new_states[index] = state_
        self.dones[index] = done
        self.rewards[index] = reward

        for agent in self.agents:
            self.actions[agent][index] = action[agent]
            self.actor_states[agent][index] = raw_obs[agent]
            self.actor_new_states[agent][index] = raw_obs_[agent]
            self.probs[agent][index] = probs[agent]
        self.mem_cntr += 1

    def clear_memory(self):
        self.states = np.zeros((self.n_states, self.n_procs, self.critic_dims),
                               dtype=np.float32)
        self.rewards = np.zeros((self.n_states, self.n_procs, self.n_agents),
                                dtype=np.float32)
        self.dones = np.zeros((self.n_states, self.n_procs), dtype=np.float32)
        self.new_states = np.zeros((self.n_states, self.n_procs,
                                   self.critic_dims), dtype=np.float32)

        self.actor_states = {a: np.zeros(
            (self.n_states, self.n_procs, self.actor_dims[a]))
                             for a in self.agents}
        self.actor_new_states = {a: np.zeros(
            (self.n_states, self.n_procs, self.actor_dims[a]))
                                 for a in self.agents}
        self.actions = {a: np.zeros(
            (self.n_states, self.n_procs, self.n_actions[a]))
                        for a in self.agents}
        self.probs = {a: np.zeros(
            (self.n_states, self.n_procs, self.n_actions[a]))
                      for a in self.agents}
