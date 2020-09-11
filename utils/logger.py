import ray
import os
import time
import torch
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
        self.writer = SummaryWriter(self.log_dir)

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
                o, r, d, i = self.env.step(a)
                reward += r
            test_rewards.append(reward)

        imgs = []
        o = self.env.reset()
        d = False
        best_reward = 0
        while not d:
            screen = self.env.render(mode='rgb_array')
            imgs.append(screen)
            a = self.actor.act(o, sample_fn=lambda dist: dist.mode)
            o, r, d, i = self.env.step(a)
            best_reward += r
        imgs = np.stack(imgs)
        imgs = torch.as_tensor(imgs).permute(0, 3, 1, 2).unsqueeze(0)

        info['reward_mean'] = np.mean(test_rewards)
        info['reward_std'] = np.std(test_rewards)
        info['best_reward'] = best_reward

        print(f'In test {self.count}:\n\treward: {info["reward_mean"]} +- {info["reward_std"]}\n\tbest_reward: {best_reward}')

        for k, v in info.items():
            self.writer.add_scalar(k, v, self.count)
        self.writer.add_video('test', imgs, self.count, fps=60)

        self.writer.flush()