
import ray
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from tianshou.data.batch import Batch

from simrl.utils import setup_seed, soft_critic_update
from simrl.utils.modules import ContinuousActor, OnehotActor, Critic, EnsembleTransition
from simrl.utils.envs import make_env
from simrl.utils.data import Collector, ReplayBuffer
from simrl.utils.logger import Logger
from simrl.utils.actor import DistributionActor, RandomActor

class MBPO:
    @staticmethod
    def get_config():
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='Pendulum-v0')
        parser.add_argument('--seed', type=int, default=None)
        parser.add_argument('--epoch', type=int, default=100)

        # parameter for transition model
        parser.add_argument('--transition_steps_per_epoch', type=int, default=200)
        parser.add_argument('--transition_batch_size', type=int, default=32)
        parser.add_argument('-thf', '--transition_hidden_feature', type=int, default=200)
        parser.add_argument('-thl', '--transition_hidden_layers', type=int, default=4)
        parser.add_argument('--ensemble_size', type=int, default=7)
        parser.add_argument('--transition_lr', type=float, default=3e-4)

        # parameter for data collection
        parser.add_argument('--initial_samples', type=int, default=5000)
        parser.add_argument('--data_collection_per_epoch', type=int, default=1000)
        parser.add_argument('--buffer_size', type=int, default=int(1e8))

        # parameter for branch rollout
        parser.add_argument('--rollout_batch_size', type=int, default=400)
        parser.add_argument('--rollout_horizon', type=int, default=1)

        # parameter for sac update
        parser.add_argument('--sac_batch_size', type=float, default=32)
        parser.add_argument('--actor_lr', type=float, default=1e-3)
        parser.add_argument('--critic_lr', type=float, default=1e-3)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--tau', type=float, default=0.005)
        parser.add_argument('--sac_update_per_step', type=int, default=10)
        parser.add_argument('--base_alpha', type=float, default=0.2)
        parser.add_argument('--auto_alpha', type=lambda x: [False, True][int(x)], default=True)
        parser.add_argument('-ahf', '--actor_hidden_features', type=int, default=256)
        parser.add_argument('-ahl', '--actor_hidden_layers', type=int, default=2)
        parser.add_argument('-aa', '--actor_activation', type=str, default='leakyrelu')
        parser.add_argument('-an', '--actor_norm', type=str, default=None)
        parser.add_argument('-chf', '--critic_hidden_features', type=int, default=256)
        parser.add_argument('-chl', '--critic_hidden_layers', type=int, default=2)
        parser.add_argument('-ca', '--critic_activation', type=str, default='leakyrelu')
        parser.add_argument('-cn', '--critic_norm', type=str, default=None)

        parser.add_argument('--test-num', type=int, default=20)
        parser.add_argument('--test_frequency', type=int, default=1)
        parser.add_argument('--log_video', action='store_true')
        parser.add_argument('--log', type=str, default=None)
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
        
        args = parser.parse_args()
        
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
            self.actor = ContinuousActor(self.state_dim, self.action_dim,
                                         hidden_features=self.config.get('actor_hidden_features', 128),
                                         hidden_layers=self.config.get('actor_hidden_layers', 1),
                                         hidden_activation=self.config.get('actor_activation', 'leakyrelu'),
                                         norm=self.config.get('actor_norm', None))            
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

        self.transition = EnsembleTransition(obs_dim=self.state_dim,
                                             action_dim=self.action_dim,
                                             hidden_features=self.config['transition_hidden_feature'],
                                             hidden_layers=self.config['transition_hidden_layers'],
                                             ensemble_size=self.config['ensemble_size'],
                                             with_reward=True)

        self.env_buffer = ReplayBuffer(self.config['buffer_size'])
        self.model_buffer = ReplayBuffer(self.config['rollout_batch_size'] * self.config['rollout_horizon'] * self.config['sac_update_per_step'])
        self.collector = Collector.remote(self.config, RandomActor(self.env.action_space))
        self.env_buffer.put(ray.get(self.collector.collect_steps.remote(self.config['initial_samples'], None)))
        ray.get(self.collector.set_actor.remote(DistributionActor(self.actor)))
        self.logger = Logger.remote(config, deepcopy(DistributionActor(self.actor)), 'mbpo')

        self.actor = self.actor.to(self.device)
        self.q1 = self.q1.to(self.device)
        self.q1_target = deepcopy(self.q1)
        self.q1_target.requires_grad_(False)
        self.q2 = self.q2.to(self.device)
        self.q2_target = deepcopy(self.q2)
        self.q2_target.requires_grad_(False)
        self.transition = self.transition.to(self.device)

        self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self.device) + np.log(self.config['base_alpha'])).float()
        if self.config['auto_alpha']:
            if self.config['env_type'] == 'discrete':
                self.target_entropy = 0.98 * np.log(self.action_dim)
            elif self.config['env_type'] == 'continuous':
                self.target_entropy = - self.action_dim
            self.alpha_optimizor = torch.optim.Adam([self.log_alpha], lr=1e-4)

        self.transition_optimizor = torch.optim.Adam(self.transition.parameters(), lr=config['transition_lr'])
        self.actor_optimizor = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizor = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=config['critic_lr'])

    def run(self):
        for i in range(self.config['epoch']):
            ''' Step 1: train transitions from current D_env '''
            for m in range(self.config['transition_steps_per_epoch']):
                batchs = self.env_buffer.sample(self.config['transition_batch_size'])
                batchs.to_torch(dtype=torch.float32, device=self.device)
                output_dist = self.transition(batchs['obs'], batchs['action'])
                nll_loss = - output_dist.log_prob(torch.cat([batchs['next_obs'], batchs['reward']], dim=-1)).mean()

                self.transition_optimizor.zero_grad()
                nll_loss.backward()
                self.transition_optimizor.step()

            for e in tqdm(range(self.config['data_collection_per_epoch'])):
                ''' Step 2: collect new samples, and add it to D_env '''
                self.env_buffer.put(ray.get(self.collector.collect_steps.remote(1, self.actor.get_weights())))

                ''' Step 3: branch rollout from uniform samples in the D_env, add to D_model '''
                batchs = self.env_buffer.sample(self.config['rollout_batch_size'])
                batchs.to_torch(dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    obs = batchs['obs']
                    for t in range(self.config['rollout_horizon']):
                        action_dist = self.actor(obs)
                        action = action_dist.sample()
                        output_dist = self.transition(obs, action)
                        output = output_dist.sample()
                        reward = output[..., -1:].mean(dim=0)
                        model_index = np.random.randint(0, output.shape[0], output.shape[1])
                        next_obs = output[model_index, np.arange(output.shape[1]), :-1]

                        batch = Batch(
                            obs=obs,
                            action=action,
                            reward=reward, 
                            done=torch.zeros_like(reward),
                            next_obs=next_obs,
                        )
                        batch.to_torch(device='cpu')

                        self.model_buffer.put(batch)

                        obs = next_obs

                ''' Step 4: perform sac update actor and critic '''
                for g in range(self.config['sac_update_per_step']):
                    batchs = self.model_buffer.sample(self.config['sac_batch_size'])
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
                        alpha_loss = - torch.mean(self.log_alpha * (log_prob + self.target_entropy).detach())

                        self.alpha_optimizor.zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizor.step()

                    q = torch.min(self.q1(batchs['obs'], new_action), self.q2(batchs['obs'], new_action))
                    actor_loss = - q.mean() + torch.exp(self.log_alpha) * log_prob.mean()

                    self.actor_optimizor.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizor.step()
                
                info = {
                    "nll" : nll_loss.item(),
                    "critic_loss" : critic_loss.item(),
                    "actor_loss" : actor_loss.item(),
                }

                self.logger.test_and_log.remote(None, info)
                
            self.logger.test_and_log.remote(self.actor.get_weights() if i % self.config['test_frequency'] == 0 else None, info)
                

if __name__ == '__main__':
    ray.init()
    config = MBPO.get_config()
    config['seed'] = config['seed'] or random.randint(0, 1000000)
    experiment = MBPO(config)
    experiment.run()
