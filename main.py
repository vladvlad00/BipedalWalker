import time

import gym
import numpy as np
from td3 import TD3
from buffer import Buffer
import matplotlib.pyplot as plt
import os


def add_noise(action, action_dim, max_action, exploration_noise=0.1):
    action = action + np.random.normal(0, exploration_noise, size=action_dim)
    action = np.clip(action, -max_action, max_action)
    return action


def train(mini_batch=False):
    lr = 0.001
    replay_batch_size = 100
    minibatch_size = 10
    gamma = 0.99
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    nr_episodes = 1000
    avg_interval = 10
    checkpoint_interval = 50

    checkpoints_root = os.path.join('checkpoints', time.strftime("%Y-%m-%d_%H-%M-%S"))
    env = gym.make('BipedalWalker-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = Buffer()

    rewards = []

    for i_episode in range(1, nr_episodes + 1):
        episode_reward = 0
        state = env.reset()
        nr_steps = 0
        while True:
            env.render()
            action = agent.select_action(state)
            action = add_noise(action, action_dim, max_action)

            next_state, reward, done, _ = env.step(action)
            nr_steps += 1
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state

            episode_reward += reward

            if done or (mini_batch and nr_steps == minibatch_size):
                agent.update(replay_buffer, nr_steps, replay_batch_size, gamma, tau, policy_noise, noise_clip,
                             policy_freq)
                nr_steps = 0
                if done:
                    break
        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-avg_interval:])
        print(f'Episode: {i_episode}\tReward: {episode_reward}\tAverage Reward (last {avg_interval}):'
              f' {avg_reward}')

        if i_episode % checkpoint_interval == 0:
            checkpoint_dir = os.path.join(checkpoints_root, 'checkpoint_' + str(i_episode))
            os.makedirs(checkpoint_dir, exist_ok=True)

            x = [i for i in range(i_episode)]
            avg_y = [np.mean(rewards[i - avg_interval:i]) for i in range(1, i_episode + 1)]
            reward_y = [i for i in rewards]
            plt.plot(x, avg_y, label="Last avg interval")
            plt.plot(x, reward_y, label="Rewards")
            plt.xlabel('nr episode')
            plt.legend()
            plt.savefig(os.path.join(checkpoint_dir, f'Plot_{i_episode}.png'))
            plt.close()

            agent.save(os.path.join(checkpoint_dir, f'checkpoint_{i_episode}'))


if __name__ == '__main__':
    train(mini_batch=True)
