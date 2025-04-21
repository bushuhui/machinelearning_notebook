import numpy as np
from mappo import MAPPO
from memory import PPOMemory
from utils import obs_list_to_state_vector
from vec_env import make_vec_envs


def run():
    env_id = 'Simple_Speaker_Listener'
    random_seed = 0
    n_procs = 2
    env = make_vec_envs(env_id, random_seed, n_procs)
    N = 2048
    batch_size = 64
    n_epochs = 10
    alpha = 3e-4
    scenario = 'simple_speaker_listener'

    n_agents = env.max_num_agents

    actor_dims = {}
    n_actions = {}
    for agent in env.agents:
        actor_dims[agent] = env.observation_space(agent).shape[0]
        n_actions[agent] = env.action_space(agent).shape[0]
    critic_dims = sum([actor_dims[a] for a in env.agents])

    mappo_agents = MAPPO(actor_dims=actor_dims, critic_dims=critic_dims,
                         n_agents=n_agents, n_actions=n_actions,
                         n_epochs=n_epochs, env=env, gamma=0.95, alpha=alpha,
                         T=N, n_procs=n_procs, scenario=scenario)

    memory = PPOMemory(batch_size, N, n_agents, env.agents,
                       n_procs, critic_dims, actor_dims, n_actions)

    MAX_STEPS = 1_000_000
    total_steps = 0
    episode = 1
    traj_length = 0
    score_history, steps_history = [], []

    while total_steps < MAX_STEPS:
        observation, _ = env.reset()
        terminal = [False] * n_procs
        score = [0] * n_procs
        while not any(terminal):
            a_p = [mappo_agents.choose_action(observation[idx]) for
                   idx in range(n_procs)]
            action = [a[0] for a in a_p]
            prob = [a[1] for a in a_p]
            print(f'action {action}')
            observation_, reward, done, trunc, info = env.step(action)

            print(f'observation_ {observation_}')
            exit()

            total_steps += 1
            traj_length += 1

            done_arr = [list(d.values()) for d in done]
            obs_arr = [list(o.values()) for o in observation]
            reward_arr = [list(r.values()) for r in reward]
            new_obs_arr = [list(o.values()) for o in observation_]
            trunc_arr = [list(t.values()) for t in trunc]

            action_dict = {agent: [list(a[agent]) for a in action]
                           for agent in env.agents}
            obs_dict = {agent: [list(o[agent]) for o in observation]
                        for agent in env.agents}
            new_obs_dict = {agent: [list(o[agent]) for o in observation_]
                            for agent in env.agents}
            probs_dict = {agent: [list(p[agent]) for p in prob]
                          for agent in env.agents}

            state = obs_list_to_state_vector(obs_arr)
            state_ = obs_list_to_state_vector(new_obs_arr)

            score += [sum(r) for r in reward_arr]

            terminal = [any(d) or any(t) for d, t in zip(done_arr, trunc_arr)]
            mask = [0.0 if t else 1.0 for t in terminal]
            memory.store_memory(obs_dict, state, action_dict,
                                probs_dict, reward_arr,
                                new_obs_dict, state_, mask)

            if traj_length % N == 0:
                mappo_agents.learn(memory)
                traj_length = 0
                memory.clear_memory()
            observation = observation_
        score_history.append(sum(score)/n_procs)
        steps_history.append(total_steps)
        avg_score = np.mean(score_history[-100:])
        print(f'{env_id} Episode {episode} total steps {total_steps}'
              f' avg score {avg_score :.1f}')

        episode += 1

    np.save('data/mappo_scores.npy', np.array(score_history))
    np.save('data/mappo_steps.npy', np.array(steps_history))
    env.close()


if __name__ == '__main__':
    run()
