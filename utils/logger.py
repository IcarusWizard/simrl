import ray
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .envs import make_env

@ray.remote
class Logger:
    def __init__(self, config, actor):
        self.config = config
        self.actor = actor
        self.count = 0

        self.env = make_env(config)

        log_name = self.config['log'] or time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
        self.log_dir = os.path.join('logs', self.config['env'], 'ppo', log_name)
        # self.writer = SummaryWriter(self.log_dir)

    def test_and_log(self, actor_state_dict, info={}):
        self.count += 1
        self.actor.load_state_dict(actor_state_dict)
        self.actor.cpu()

        test_rewards = []
        for _ in range(self.config['test_num']):
            reward = 0
            o = self.env.reset()
            d = False
            while not d:
                a = self.actor.act(o)
                o, r, d, info = self.env.step(a)
                reward += r
            test_rewards.append(reward)

        info['reward_mean'] = np.mean(test_rewards)
        info['reward_std'] = np.std(test_rewards)

        print(f'In test {self.count}, reward: {info["reward_mean"]} +- {info["reward_std"]}')