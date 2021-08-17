"""
    This file is from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import gym
from gym import wrappers

from util.seed import set_global_seeds
from util.atari_wrapper import wrap_deepmind, wrap_deepmind_ram

def get_env(game_name, seed, idx_to_save_video=tuple()):
    env = gym.make(game_name)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = './videos'
    env = wrappers.Monitor(env, expt_dir, force=True, video_callable=lambda x: (x in idx_to_save_video))
    env = wrap_deepmind(env)

    return env

def get_ram_env(env, seed):
    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/gym-results'
    env = wrappers.Monitor(env, expt_dir, force=True)
    env = wrap_deepmind_ram(env)

    return env

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)
