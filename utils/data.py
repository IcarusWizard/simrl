import ray
import numpy as np
from tianshou.data import Batch

from .envs import make_env

@ray.remote
class Collector:
    def __init__(self, config, actor):
        self.config = config
        self.env = make_env(self.config)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.actor = actor
        self.o = self.env.reset()

        self.print_env_info()

    def collect_steps(self, steps, actor_state_dict):
        self.actor.load_state_dict(actor_state_dict)
        self.actor.cpu()

        batchs = []

        for _ in range(steps):
            a = self.actor.act(self.o)
            n_o, r, d, info = self.env.step(a)

            batchs.append(Batch(
                obs=self.o,
                next_obs=n_o,
                reward=[r],
                action=a,
                done=[float(d)],
            ))

            if d:
                self.o = self.env.reset()
            else:
                self.o = n_o

        batchs = Batch.stack(batchs)

        return batchs

    def print_env_info(self):
        print('env name:', self.config['env'])
        print('env type:', self.config['env_type'])
        print('state dim:', self.state_dim)
        print('action dim:', self.action_dim)
        