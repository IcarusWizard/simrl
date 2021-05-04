import os
import gym
import random
import numpy as np

def make_env(config) -> gym.Env:
    env_name = config['env']
    env = gym.make(env_name)

    if type(env.action_space) is gym.spaces.Discrete:
        env = OneHotAction(env)
        config['env_type'] = 'discrete'
    else:
        config['env_type'] = 'continuous'
        config['max_action'] = env.action_space.high
        config['min_action'] = env.action_space.low

    return env

class ActionRepeat:
    def __init__(self, env, amount):
        self._env = env
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info

class OneHotSpace(gym.Space):
    def sample(self):
        actions = self.shape[0]
        index = random.randint(0, actions - 1)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference

    def contains(self, x):
        return all(x >= 0 & x <= 1) and x.sum() == 1.0

class OneHotAction:
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    @property
    def action_space(self):
        shape = (self._env.action_space.n,)
        space = OneHotSpace(shape=shape, dtype=np.float32)
        return space

    def step(self, action):
        index = np.argmax(action).astype(int)
        return self._env.step(index)

    def reset(self):
        return self._env.reset()