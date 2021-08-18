import numpy as np
import random
import gym.spaces
from itertools import count
from tqdm import tqdm
from matplotlib import pyplot as plt

from model.dqn import DQN
from model.dqn import ReplayBuffer, select_epilson_greedy_action
from util.schedule import LinearSchedule
from util.gym import get_env, get_wrapper_by_name

import torch
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
env_name = "SpaceInvadersNoFrameskip-v0"
env = get_env(env_name, 0)
replay_buffer_size=1000000
batch_size=32
gamma=0.99
learning_starts=50000
learning_freq=4
frame_history_len=4
target_update_freq=10000
expname=None
total_timestep=1_000_000
evaluation_timestep = 500
exploration = LinearSchedule(total_timestep, 0.1)
log_every_n_steps = 10000
learning_rate=0.00025
alpha=0.95
eps=0.01
device = "cuda" if torch.cuda.is_available() else "cpu"

# Build model
img_h, img_w, img_c = env.observation_space.shape
input_arg = frame_history_len * img_c
num_actions = env.action_space.n

# Initialize target q function and q function
policy = DQN(input_arg, num_actions).to(device)
target = DQN(input_arg, num_actions).to(device)

# Construct Q network optimizer function
criterion = torch.nn.MSELoss(reduction='none')
optimizer = optim.RMSprop(policy.parameters(), lr=learning_rate, alpha=alpha, eps=eps)

# Construct the replay buffer
replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
validation_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

# Fill Validation Buffer
done = True
val_env = get_env(env_name, 0)
for t in count():
    if t > evaluation_timestep:
        break
    if done:
        obs, done = val_env.reset(), False
    last_idx = validation_buffer.store_frame(obs)
    obs, _, done, _ = val_env.step(np.random.randint(0, num_actions))
    validation_buffer.store_effect(last_idx, 0, 0, done)

# Run env
num_param_updates = 0
mean_episode_reward = -float('nan')
best_mean_episode_reward = -float('inf')

# Logger
eval_reward = -float('inf')
eval_steps, eval_rewards, eval_q_values = [], [], []

# Main loop
last_obs = env.reset()
with tqdm(total=total_timestep) as pbar:
    for t in count():
        pbar.n = t
        pbar.refresh()

        if t >= total_timestep:
            break

        last_idx, prev_obs = replay_buffer.store_frame(last_obs), replay_buffer.encode_recent_observation()
        action = select_epilson_greedy_action(policy, prev_obs, exploration.value(t)) if t > learning_starts else random.randrange(num_actions)
        curr_obs, reward, done, _ = env.step(action)
        reward = max(-1.0, min(reward, 1.0)) # clip rewards between -1 and 1
        replay_buffer.store_effect(last_idx, action, reward, done)

        if done:
            curr_obs = env.reset()
        last_obs = curr_obs

        # Train network.
        if (t > learning_starts and
            t % learning_freq == 0 and
            replay_buffer.can_sample(batch_size)):

            curr_obs_batch, action_batch, reward_batch, next_obs_batch, not_done_mask = replay_buffer.sample(batch_size)
            # done_mask[i] is 1 if the next state is the end of an episode, so no Q-value at the next state

            current_Q_values = policy(curr_obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
            next_Q_values = target(next_obs_batch).detach().max(1)[0] * not_done_mask
            target_Q_values = reward_batch + (gamma * next_Q_values)
            bellman_error = criterion(current_Q_values, target_Q_values)

            optimizer.zero_grad()
            bellman_error = torch.clamp(bellman_error, 0, 1).sum()
            bellman_error.backward()
            num_param_updates += 1
            optimizer.step()

            if num_param_updates % target_update_freq == 0:
                target.load_state_dict(policy.state_dict())

        # Logger
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 100:
            mean_episode_reward = np.mean(episode_rewards[-100:])
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        # Evaluation
        if t % log_every_n_steps == 0 and t > learning_starts:
            policy.eval()

            tqdm.write("===============Training==================")
            tqdm.write(f"timestep {t} & episodes {len(episode_rewards)}")
            tqdm.write(f"mean reward (100 episodes) {mean_episode_reward}")
            tqdm.write(f"best mean reward {best_mean_episode_reward}")
            #tqdm.write(f"exploration {exploration.value(t)}")

            # Test Q-values over validation memory
            qs = []
            with torch.no_grad():
                for state in validation_buffer.get_all_frames():  # Iterate over valid states
                    state = (torch.from_numpy(state).type(torch.cuda.FloatTensor) / 255.0)[None, ...]
                    qs.append(policy(state).max(1)[0].item())
            eval_reward = max(np.mean(episode_rewards[-5:]), eval_reward)
            eval_steps.append(t)
            eval_rewards.append(eval_reward)
            eval_q_values.append(np.mean(qs))
            tqdm.write("===============Validation=================")
            tqdm.write(f"Rewards {eval_rewards[-1]}")
            tqdm.write(f"steps {eval_steps[-1]}")
            tqdm.write(f"mean Q-values {eval_q_values[-1]}")
            tqdm.write("\n")

            policy.train()

torch.save(policy.state_dict(), f'./result/policy_{t:06}.pt')
plt.figure(figsize=(15, 15))
plt.subplot(2, 1, 1)
plt.title('reward')
plt.plot(eval_steps, eval_rewards, 'r')
plt.subplot(2, 1, 2)
plt.title('q values')
plt.plot(eval_steps, eval_q_values, 'b')
plt.tick_params(axis='both', direction='in', length=3, pad=6, labelsize=14)
plt.savefig('./result/metrics.png')

'''
policy.eval()
vid_env = get_env(env_name, 0, (0,))
vid_env.reset()
done = False
tmp_buf = ReplayBuffer(10000, frame_history_len)
for t in count():
    if done:
        break
    last_idx = tmp_buf.store_frame(obs)
    recent_observations = replay_buffer.encode_recent_observation()
    action = select_epilson_greedy_action(policy, recent_observations, 0.0)
    obs, _, done, _ = vid_env.step(action)
    tmp_buf.store_effect(last_idx, 0, 0, done)
'''
