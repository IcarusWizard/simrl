import ray
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from simrl.utils import setup_seed, soft_critic_update
from simrl.utils.modules import OnehotActor, BoundedContinuousActor, Critic
from simrl.utils.envs import make_env
from simrl.utils.data import CollectorServer, ReplayBuffer
from simrl.utils.logger import Logger

class SAC:
    @staticmethod
    def get_config():
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='CartPole-v1')
        parser.add_argument('--seed', type=int, default=None)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--buffer_size', type=int, default=int(1e6))
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--tau', type=float, default=0.005)
        parser.add_argument('--epoch', type=int, default=100)
        parser.add_argument('--data_collection_per_epoch', type=int, default=10000)
        parser.add_argument('--num_collectors', type=int, default=4)
        parser.add_argument('--training_step_per_epoch', type=int, default=1000)
        parser.add_argument('--test-num', type=int, default=20)
        parser.add_argument('--base_alpha', type=float, default=0.2)
        parser.add_argument('--auto_alpha', type=lambda x: [False, True][int(x)], default=True)
        parser.add_argument('--test_frequency', type=int, default=5)
        parser.add_argument('--log_video', action='store_true')
        parser.add_argument('--log', type=str, default=None)
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

        parser.add_argument('-ahf', '--actor_hidden_features', type=int, default=128)
        parser.add_argument('-ahl', '--actor_hidden_layers', type=int, default=2)
        parser.add_argument('-aa', '--actor_activation', type=str, default='leakyrelu')
        parser.add_argument('-an', '--actor_norm', type=str, default=None)
        parser.add_argument('-chf', '--critic_hidden_features', type=int, default=128)
        parser.add_argument('-chl', '--critic_hidden_layers', type=int, default=2)
        parser.add_argument('-ca', '--critic_activation', type=str, default='leakyrelu')
        parser.add_argument('-cn', '--critic_norm', type=str, default=None)
        
        args = parser.parse_known_args()[0]
        
        return args.__dict__

    def __init__(self, config):
        self.config = config
        setup_seed(self.config['seed'])
        self.env = make_env(config)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.device = self.config['device']

        if self.config['env_type'] == 'discrete':
            self.actor = OnehotActor(self.state_dim, self.action_dim,
                                     hidden_features=self.config.get('actor_hidden_features', 128),
                                     hidden_layers=self.config.get('actor_hidden_layers', 1),
                                     hidden_activation=self.config.get('actor_activation', 'leakyrelu'),
                                     norm=self.config.get('actor_norm', None))
        elif self.config['env_type'] == 'continuous':
            self.actor = BoundedContinuousActor(self.state_dim, self.action_dim,
                                                hidden_features=self.config.get('actor_hidden_features', 128),
                                                hidden_layers=self.config.get('actor_hidden_layers', 1),
                                                hidden_activation=self.config.get('actor_activation', 'leakyrelu'),
                                                norm=self.config.get('actor_norm', None),
                                                min_action=self.config.get('min_action', -1),
                                                max_action=self.config.get('max_action', 1))            
        else:
            raise ValueError('{} is not supported!'.format(self.config['env_type']))

        self.q1 = Critic(self.state_dim,
                         action_dim=self.action_dim,
                         hidden_features=self.config.get('critic_hidden_features', 128),
                         hidden_layers=self.config.get('critic_hidden_layers', 1),
                         hidden_activation=self.config.get('critic_activation', 'leakyrelu'),
                         norm=self.config.get('critic_norm', None))

        self.q2 = Critic(self.state_dim,
                         action_dim=self.action_dim,
                         hidden_features=self.config.get('critic_hidden_features', 128),
                         hidden_layers=self.config.get('critic_hidden_layers', 1),
                         hidden_activation=self.config.get('critic_activation', 'leakyrelu'),
                         norm=self.config.get('critic_norm', None))

        self.buffer = ray.remote(ReplayBuffer).remote(self.config['buffer_size'])
        self.collector = CollectorServer.remote(self.config, deepcopy(self.actor), self.buffer, self.config['num_collectors'])
        self.logger = Logger.remote(config, deepcopy(self.actor), 'sac')

        self.actor = self.actor.to(self.device)
        self.q1 = self.q1.to(self.device)
        self.q2 = self.q2.to(self.device)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2) 

        self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self.device) + np.log(self.config['base_alpha'])).float()
        if self.config['auto_alpha']:
            if self.config['env_type'] == 'discrete':
                self.target_entropy = np.log(self.action_dim)
            elif self.config['env_type'] == 'continuous':
                self.target_entropy = - self.action_dim
            self.alpha_optimizor = torch.optim.Adam([self.log_alpha], lr=1e-3)

        self.actor_optimizor = torch.optim.Adam(self.actor.parameters(), lr=config['lr'])
        self.critic_optimizor = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=config['lr'])

    def run(self):
        for i in tqdm(range(self.config['epoch'])):
            batchs_id = self.collector.collect_steps.remote(self.config['data_collection_per_epoch'], self.actor.get_weights())
            batchs = ray.get(batchs_id)
        
            for _ in tqdm(range(self.config['training_step_per_epoch'])):
                batchs = ray.get(self.buffer.sample.remote(self.config['batch_size']))

                batchs.to_torch(dtype=torch.float32, device=self.device)

                ''' update critic '''
                with torch.no_grad():
                    next_action_dist = self.actor(batchs['next_obs'])
                    next_action = next_action_dist.mode
                    next_action_log_prob = next_action_dist.log_prob(next_action)
                    next_q1 = self.q1_target(batchs['next_obs'], next_action)
                    next_q2 = self.q2_target(batchs['next_obs'], next_action)
                    next_q = torch.min(next_q1, next_q2)
                    target = batchs['reward'] + self.config['gamma'] * (1 - batchs['done']) * (next_q - next_action_log_prob.unsqueeze(dim=-1))

                q1 = self.q1(batchs['obs'], batchs['action'])
                q2 = self.q2(batchs['obs'], batchs['action'])
                critic_loss = torch.mean((q1 - target) ** 2) + torch.mean((q2 - target) ** 2)

                self.critic_optimizor.zero_grad()
                critic_loss.backward()
                self.critic_optimizor.step()

                soft_critic_update(self.q1, self.q1_target, self.config['tau'])
                soft_critic_update(self.q2, self.q2_target, self.config['tau'])

                ''' update actor '''
                action_dist = self.actor(batchs['obs'])
                new_action = action_dist.rsample()
                log_prob = action_dist.log_prob(new_action)

                if self.config['auto_alpha']:
                    # update alpha
                    alpha_loss = - torch.mean(self.log_alpha * (log_prob + self.target_entropy).detach())

                    self.alpha_optimizor.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizor.step()

                # update actor
                q = torch.min(self.q1(batchs['obs'], new_action), self.q2(batchs['obs'], new_action))
                actor_loss = - q.mean() + torch.exp(self.log_alpha) * log_prob.mean()

                self.actor_optimizor.zero_grad()
                actor_loss.backward()
                self.actor_optimizor.step()

                info = {
                    "critic_loss" : critic_loss.item(),
                    "actor_loss" : actor_loss.item(),
                }

                self.logger.test_and_log.remote(None, info)

            self.logger.test_and_log.remote(self.actor.get_weights() if i % self.config['test_frequency'] == 0 else None, info)

if __name__ == '__main__':
    ray.init()
    config = SAC.get_config()
    config['seed'] = config['seed'] or random.randint(0, 1000000)
    experiment = SAC(config)
    experiment.run()
