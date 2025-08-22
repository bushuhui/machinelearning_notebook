#!/bin/bash python3
# last update:2022-6-7
# performance is not good, consider change the algorithm parameters

import gym
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple, deque
from itertools import count
from PIL import Image


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

import os

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):
    """
    DQN replay buffer to store deque of transitions
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """
    DQN algorithm implementation
    """

    def __init__(self, h, w, outputs, device):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # get linear input size from input image size
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        # get size by 3 times due to 3 layers of conv
        convw = conv2d_size_out(conv2d_size_out((conv2d_size_out(w))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        self.device = device

    # returns tensor([[left0exp,right0exp]...])
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))  # ?


# image input process
resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])


def get_cart_location(env, screen_width):
    """
    get cart location by env state
    """
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def get_screen(env):
    # returned screen is 400x600x3 for gym normally, and transpose it to torch order(CHW)
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    # strip the top and bottom of screen due to cart pos
    screen = screen[:, int(screen_height*0.4): int(screen_height*0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        # left
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        # right
        slice_range = slice(-view_width, None)
    else:
        # middle
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)

    # get square image centered on cart
    screen = screen[:, :, slice_range]
    # convert to float, rescale and convert to torch tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255  # 255 for normalize
    screen = torch.from_numpy(screen)
    # add batch dimension to get BCHW

    return resize(screen).unsqueeze(0)


def plt_screen_extract(env):
    """
    plot an example screen extracted and resized from gym env
    """
    env.reset()
    plt.figure()
    plt.imshow(get_screen(env).cpu().squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
    plt.pause(5)


def select_action(policy_net, state, steps_done, device, n_actions, eps_s, eps_e, eps_d):
    """
    select certain action from state input by epsilon-strategy and net prediction
    """
    sample = random.random()
    eps_threhold = eps_e + (eps_s - eps_e) * math.exp(-1.*steps_done/eps_d)
    steps_done += 1
    if sample > eps_threhold:
        # get action by dqn net
        with torch.no_grad():
            action =  policy_net(state).max(1)[1].view(1, 1)
    else:
        # get action by random
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    return action, steps_done


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # take 100 episodes average to plot too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(00), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause for update


def optimize_model(policy_net, target_net, optimizer, criterion, memory, batch_size, device, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # transpose like: [('a', 1), ('b', 2)] -> [('a', 'b'), (1, 2)]
    batch = Transition(*zip(*transitions))
    # compute the mask of states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # compute Q(s_t,a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # compute V(s_{t+1}) for all states and get their best reward with max(1)[0]
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # compute expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    # compute Huber loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # optimization
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train_loop(env, policy_net, target_net, optimizer, criterion, memory, device,
               n_actions, episodes, batch_size, gamma, eps_s, eps_e, eps_d, target_update):
    steps_done = 0
    episode_durations = []
    for episode in range(episodes):
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        for t in count():
            # get and perform an action
            action, steps_done = select_action(policy_net=policy_net,
                                               state=state,
                                               steps_done=steps_done,
                                               device=device,
                                               n_actions=n_actions,
                                               eps_s=eps_s,
                                               eps_e=eps_e,
                                               eps_d=eps_d)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            # observe new state
            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None
            # store transition into memory
            memory.push(state, action, next_state, reward)
            # move to next state
            state = next_state
            # perform optimization on policy net
            optimize_model(policy_net=policy_net,
                           target_net=target_net,
                           optimizer=optimizer,
                           criterion=criterion,
                           memory=memory,
                           batch_size=batch_size,
                           device=device,
                           gamma=gamma)
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
        if episode % target_update == 0:
            print("---update target net----")
            target_net.load_state_dict(policy_net.state_dict())

    print("---complete---")
    env.render()
    env.close()
    plt.ioff()
    plt.show()


def train_dqn_cartpole():
    # env make up
    env = gym.make("CartPole-v0").unwrapped
    # plot set-up
    plt.ion()
    # device set-up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # see example screen
    plt_screen_extract(env)
    # hyperparameters set-up
    BATCH_SIZE = 128
    GAMMA = 0.999   # discount
    EPS_START = 0.9  # greedy strategy factor epsilon
    EPS_END = 0.05
    EPS_DECAY = 200     # epsilon decay over step
    TARGET_UPDATE = 10
    episodes = 500

    # get screen with parameters to initialize dqn layers
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape
    # get action number
    n_actions = env.action_space.n

    policy_net = DQN(screen_height, screen_width, n_actions, device).to(device)
    target_net = DQN(screen_height, screen_width, n_actions, device).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()   # set to be validation mode
    optimizer = optim.RMSprop(policy_net.parameters())
    criterion = nn.SmoothL1Loss()
    memory = ReplayBuffer(int(1e4))

    print("--begin train--")
    train_loop(env=env,
               policy_net=policy_net,
               target_net=target_net,
               optimizer=optimizer,
               criterion=criterion,
               memory=memory,
               device=device,
               n_actions=n_actions,
               episodes=episodes,
               batch_size=BATCH_SIZE,
               gamma=GAMMA,
               eps_s=EPS_START,
               eps_e=EPS_END,
               eps_d=EPS_DECAY,
               target_update=TARGET_UPDATE)


train_dqn_cartpole()
print("--train finished--")
