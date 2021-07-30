import gym
import random

class Policy:
    def __init__(self, num_action):
        self.num_action = num_action

    def __call__(self):
        return random.randrange(self.num_action)

env = gym.make('Breakout-v0')
episode_reward = []
episode_length = []

policy = Policy(env.action_space.n)

num_epoch = 10
for i in range(num_epoch):
    env.reset()

    episode_reward.append(0.0)
    episode_length.append(0)
    while True:
        env.render()
        action = policy()
        state, reward, done, info = env.step(action)

        episode_reward[-1] += reward
        episode_length[-1] += 1

        if done:
            print(f"[{i:03}] episode_reward: {episode_reward[-1]}, episode_length: {episode_length[-1]}")
            break
print(f"*** mean reward of {num_epoch} episodes: {sum(episode_reward)/len(episode_reward)}, mean length: {sum(episode_length)/len(episode_reward)}")
env.close()
