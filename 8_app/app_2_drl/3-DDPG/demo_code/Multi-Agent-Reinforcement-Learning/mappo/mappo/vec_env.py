from mpi4py import MPI
import multiprocessing as mp
import gym
import numpy as np
import torch as T
from pettingzoo.mpe import simple_speaker_listener_v4


# based on:
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# and:
# https://github.com/maximecb/gym-miniworld/blob/master/pytorch-a2c-ppo-acktr/vec_env/subproc_vec_env.py

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, trunc, info = env.step(data)
            remote.send((ob, reward, done, trunc, info))
        elif cmd == 'reset':
            ob, info = env.reset()
            remote.send((ob, info))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'max_num_agents':
            remote.send(env.max_num_agents)
        elif cmd == 'agents':
            remote.send(env.agents)
        else:
            raise NotImplementedError


class SubprocVecEnv:
    def __init__(self, env_fns, spaces=None):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        mp.set_start_method('forkserver')
        self.remotes, self.work_remotes = zip(*[mp.Pipe()
                                                for _ in range(nenvs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote,
                              CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in
                   zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

        self.remotes[0].send(('reset', None))
        _, _ = self.remotes[0].recv()

        self.remotes[0].send(('max_num_agents', None))
        self.max_num_agents = self.remotes[0].recv()
        self.remotes[0].send(('agents', None))
        self.agents = self.remotes[0].recv()

    def step_async(self, actions):
        assert not self.closed, "trying to operate after calling close()"
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        # self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, truncs, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), \
               np.stack(truncs), infos
    """
    def step_wait(self):
        assert not self.closed, "trying to operate after calling close()"
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos
    """
    def reset(self):
        assert not self.closed, "trying to operate after calling close()"
        for remote in self.remotes:
            remote.send(('reset', None))
        obs_arr, info_arr = [], []
        for remote in self.remotes:
            obs, info = remote.recv()
            obs_arr.append(obs)
            info_arr.append(info)
        return np.array(obs_arr), np.array(info_arr)

    def close_extras(self):
        if self.closed:
            return
        """
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        """
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def step(self, actions):
        # self.step_async(actions)
        obs, reward, dones, truncs, info = self.step_async(actions)
        return obs, reward, dones, truncs, info
        # return self.step_wait()

    def __del__(self):
        if not self.closed:
            self.close()


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def make_env(env_id, seed, rank):
    def _thunk():
        env = simple_speaker_listener_v4.parallel_env(
            continuous_actions=True)
        _, _ = env.reset(seed=seed+rank)
        # env.seed(seed + rank)
        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes):
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    set_global_seeds(seed)
    envs = [make_env(env_name, seed, i) for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)

    return envs


def set_global_seeds(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
    T.manual_seed(seed)
