import os
import cv2
import gym
import ray
import torch
import numpy as np
from copy import deepcopy
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

from .envs import make_env
from .actor import Actor

@ray.remote
def test_on_env(env : gym.Env, actor : Actor):
    env = deepcopy(env)
    actor = deepcopy(actor)
    reward = 0
    actor.reset()
    o = env.reset()
    d = False
    while not d:
        a = actor.act(o)
        o, r, d, i = env.step(a)
        reward += r
    return reward

@ray.remote
class Logger:
    def __init__(self, config : Dict[str, Any], actor : Actor, algo_name : str):
        self.config = config
        self.actor = actor
        self.count = 0

        self.env = make_env(config)

        log_name = self.config['log'] or str(self.config['seed'])
        self.log_dir = os.path.join('logs', self.config['env'], algo_name, log_name)
        self.writer = SummaryWriter(self.log_dir)

        self.metrics = []

    def test_and_log(self,
                     parameters : Optional[Dict[str, torch.Tensor]] = None, 
                     info : Dict[str, float] = {},
                     steps : Optional[int] = None):

        self.count += 1
        steps = steps or self.count
        
        if parameters:
            self.actor.set_parameters(parameters)

            test_rewards = ray.get([test_on_env.remote(self.env, self.actor) for _ in range(self.config['test_num'])])

            if self.config['log_video']:
                try:
                    imgs = []
                    self.actor.reset()
                    o = self.env.reset()
                    d = False
                    while not d:
                        screen = self.env.render(mode='rgb_array')
                        screen = cv2.resize(screen, (128, 128))
                        imgs.append(screen)
                        a = self.actor.act(o)
                        o, r, d, i = self.env.step(a)
                    imgs = np.stack(imgs)
                    imgs = torch.as_tensor(imgs).permute(0, 3, 1, 2).unsqueeze(0)
                    self.writer.add_video('test', imgs, steps, fps=60)
                except Exception as e:
                    print(f'when logging video: {e}')

            info['reward_mean'] = np.mean(test_rewards)
            info['reward_std'] = np.std(test_rewards)

            print(f'In steps {steps}:\n\treward: {info["reward_mean"]} +- {info["reward_std"]}')
            
            self.metrics.append([steps, info['reward_mean'], info['reward_std']])
            np.savetxt(os.path.join(self.log_dir, 'metrics.txt'), self.metrics, fmt='%.3f')

        for k, v in info.items():
            self.writer.add_scalar(k, v, steps)
        
        self.writer.flush()