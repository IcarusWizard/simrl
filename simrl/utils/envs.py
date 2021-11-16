import os
import gym
import random
import warnings
import numpy as np

try:
    import pybullet_envs
except:
    warnings.warn('pybullet is not installed!')

def make_env(config) -> gym.Env:
    env_name = config['env']
    if env_name.startswith('dmc-'):
        env = DeepMindControl(env_name[4:])
    else:
        env = gym.make(env_name)

    if type(env.action_space) is gym.spaces.Discrete:
        env = OneHotAction(env)
        config['env_type'] = 'discrete'
    else:
        config['env_type'] = 'continuous'
        config['max_action'] = env.action_space.high
        config['min_action'] = env.action_space.low

    return env

class DeepMindControl:
    ''' Provide a gym like interface for dm_control '''
    def __init__(self, name : str):
        os.environ['MUJOCO_GL'] = 'egl' # use egl as backend, may lead to problems
        split_name = name.split('-')
        self.image_input = len(split_name) == 3 and split_name[-1] == 'image'
        domain, task = split_name[:2]
        from dm_control import suite
        self._env = suite.load(domain, task)
        self.size = (64, 64)
        self.camera = dict(quadruped=2).get(domain, 0)

    @property
    def observation_space(self):
        if self.image_input:
            return gym.spaces.Box(0, 255, self.size + (3,), dtype=np.uint8)
        else:
            size = 0
            for key, value in self._env.observation_spec().items():
                size += 1 if len(value.shape) == 0 else value.shape[0]
            return gym.spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        obs = self.get_obs(time_step)
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32), 'time_step' : time_step}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        return self.get_obs(time_step)

    def get_obs(self, time_step):
        if self.image_input:
            return self.render()
        else:
            return np.concatenate([v if isinstance(v, np.ndarray) else np.array([v]) for v in time_step.observation.values()])

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self.size, camera_id=self.camera)

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

    def all(self):
        references = []
        actions = self.shape[0]
        for index in range(actions):
            reference = np.zeros(actions, dtype=np.float32)
            reference[index] = 1.0
            references.append(reference)
        return references

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