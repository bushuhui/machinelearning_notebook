from agent import Agent


class MAPPO:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 env, T, n_procs, n_epochs,
                 alpha=1e-4, gamma=0.95, chkpt_dir='tmp/mappo/',
                 scenario='co-op_navigation'):
        self.agents = []
        chkpt_dir += scenario
        for agent_idx, agent in enumerate(env.agents):
            self.agents.append(Agent(actor_dims[agent], critic_dims,
                               n_actions[agent], agent_idx,
                               alpha=alpha, chkpt_dir=chkpt_dir,
                               gamma=gamma, agent_name=agent,
                               scenario=scenario))

    def save_checkpoint(self):
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = {}
        probs = {}
        for agent_id, agent in zip(raw_obs, self.agents):
            action, prob = agent.choose_action(raw_obs[agent_id])
            actions[agent_id] = action
            probs[agent_id] = prob
        return actions, probs

    def learn(self, memory):
        for agent in self.agents:
            agent.learn(memory)
