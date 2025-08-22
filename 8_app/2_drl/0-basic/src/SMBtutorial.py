# last update 2022-6-7
# a super mario bros tutorial implementation

import torch
from torch import nn
from torchvision import transforms as T

import numpy as np
from PIL import Image
from pathlib import Path
from collections import deque
import random, datetime, time, os, copy
import matplotlib.pyplot as plt

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros


# wrappers are used to preprocess the env before sending data to agent
class SkipFrame(gym.Wrapper):
    """
    skip several frames in consecutive since they have nearly the same information
    """
    def __init__(self, env, skip):
        """
        Return only every 'skip'-th frame
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """
        let env step for 'skip' steps and accumulate their reward;
        maintain the same IO interface of action as input and obs,reward,done,info as output.
        """
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:  # if terminate state reached during skipping
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    """
    remove useless color information for learning;
    turn a 3 channel image into 1 channel gray image;
    (3, h, w) -> (1, h, w)
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    @staticmethod
    def permute_orientation(observation):
        """[H,W,C] -> [C, H, W] tensor"""
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    """
    resize the observation to shape
    """
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            # make up if only an integer is given
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # resize and normalize value into range of (0,255)
        transforms = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])
        observation = transforms(observation).squeeze(0)
        return observation


# base Mario
class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1  # FIXME: what about even greedier?
        self.curr_step = 0

        self.save_every = 5e5  # save period

    def act(self, state):
        """
        choose action by state and update value of step
        Input: state(LazyFrame) with state_dim
        Output: action_idx(int)
        """
        # explore
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # exploit
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay  # FIXME: is it too frequent to drop the rate?
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return action_idx


# Mario for experience replay
class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.memory = deque(maxlen=35000)  # FIXME: GPU out of memory
        self.batch_size = 32  # FIXME: how to set batch size?

    def cache(self, state, next_state, action, reward, done):
        """
        Replay Buffer
        state: LazyFrame
        next_state: LazyFrame
        action: int
        reward: float
        done: bool
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def memory_used(self):
        return len(self.memory)


class MarioNet(nn.Module):
    """
    using DDQN algorithm
    structure: Input -> (conv2d, relu)*3 -> flatten -> (dense + relu)*2 -> Output
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # target net are frozen
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, inputs, model):
        if model == "online":
            return self.online(inputs)
        elif model == "target":
            return self.target(inputs)


class Mario(Mario):
    """
    td process of getting q values
    """
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.gamma = 0.9  # FIXME: put hyperparameters all together

    def td_estimate(self, state, action):
        current_q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
        return current_q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_q, axis=1)
        next_q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float())*self.gamma*next_q).float()


class Mario(Mario):
    """
    model parameters updating
    """
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.optimzer = torch.optim.Adam(self.net.parameters(), lr=0.00025)  # FIXME:put together
        self.loss_fn = torch.nn.SmoothL1Loss()

    def update_q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimzer.zero_grad()
        loss.backward()
        self.optimzer.step()
        return loss.item()

    def sync_q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


class Mario(Mario):
    """
    save checkpoint
    """
    def save(self):
        save_path = (self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path,)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


class Mario(Mario):
    """
    learning main loop
    """
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.burnin = 1e4  # minimum experiences before training
        self.learn_every = 3  # number of experiences between updates to q_online
        self.sync_every = 1e4  # number of experiences between sync to q_target

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_q_target()
            print(f"Target net synced at {self.curr_step} steps")
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None

        # experience replay
        state, next_state, action, reward, done = self.recall()
        # get td estimate
        td_est = self.td_estimate(state, action)
        # get td target
        td_tgt = self.td_target(reward, next_state, done)
        # back prop loss
        loss = self.update_q_online(td_est, td_tgt)

        return td_est.mean().item(), loss


class MetricLogger:
    """
    log
    """
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # current episode metric
        self.init_episode()

        # timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        """mark end of episode"""
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time-last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()


def main():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    # limit action space to walk right and jump right
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    # combine several frames into 1 point to give local time-continuity
    env = FrameStack(env, num_stack=4)

    # by now we have [4, 84, 84] as env reaction to mario's actions

    use_cuda = torch.cuda.is_available()
    print(f"Using cuda: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, 84, 84),
                  action_dim=env.action_space.n,
                  save_dir=save_dir)

    logger = MetricLogger(save_dir)

    episodes = 50000

    for e in range(episodes):
        state = env.reset()
        while True:
            # run agent on the state
            action = mario.act(state)
            # do action
            next_state, reward, done, info = env.step(action)
            # save to replay buffer
            mario.cache(state, next_state, action, reward, done)
            # learn loop
            q, loss = mario.learn()
            # log
            logger.log_step(reward, loss, q)
            # update state
            state = next_state
            # whether terminate
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
            print(f"memory used: {mario.memory_used()}")


main()
