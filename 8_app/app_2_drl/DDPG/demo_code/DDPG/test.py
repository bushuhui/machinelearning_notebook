import gymnasium
import imageio
import argparse
from DDPG import DDPG
from utils import scale_action

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='./output_images/LunarLander.gif')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/DDPG/')
parser.add_argument('--step', type=int, default=1000)
parser.add_argument('--save_video', type=bool, default=True)
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--render', type=bool, default=True)

args = parser.parse_args()


def main():
    env = gymnasium.make('LunarLanderContinuous-v3', render_mode="rgb_array")
    agent = DDPG(alpha=0.0003, beta=0.0003, 
                 state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.shape[0], 
                 actor_fc1_dim=400, actor_fc2_dim=300,
                 critic_fc1_dim=400, critic_fc2_dim=300, 
                 ckpt_dir=args.checkpoint_dir,
                 batch_size=256)
    agent.load_models(args.step)
    video = imageio.get_writer(args.filename, fps=args.fps)

    done = False
    observation = env.reset()[0]
    while not done:
        if args.render:
            env.render()
        action = agent.choose_action(observation, train=False)
        action_ = scale_action(action.copy(), env.action_space.high, env.action_space.low)
        observation_, reward, done, info, _ = env.step(action_)
        observation = observation_
        if args.save_video:
            video.append_data(env.render())


if __name__ == '__main__':
    main()

