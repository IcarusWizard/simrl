import ray
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from simrl.utils import setup_seed
from simrl.utils.modules import EnsembleTransition
from simrl.utils.envs import make_env
from simrl.utils.data import CollectorServer, ReplayBuffer
from simrl.utils.logger import Logger
from simrl.utils.actor import CEMActor

class PETS:
    @staticmethod
    def get_config():
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='Pedulum-v0')
        parser.add_argument('--seed', type=int, default=None)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--buffer_size', type=int, default=int(1e6))
        parser.add_argument('--epoch', type=int, default=100)
        parser.add_argument('--horizon', type=int, default=10)
        parser.add_argument('--samples', type=int, default=100)
        parser.add_argument('--elites', type=int, default=20)
        parser.add_argument('--iterations', type=int, default=10)
        parser.add_argument('--data_collection_per_epoch', type=int, default=100)
        parser.add_argument('--train_steps_per_epoch', type=int, default=100)
        parser.add_argument('--num_collectors', type=int, default=1)
        parser.add_argument('--test-num', type=int, default=10)
        parser.add_argument('--test_frequency', type=int, default=5)
        parser.add_argument('--log_video', action='store_true')
        parser.add_argument('--log', type=str, default=None)
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

        parser.add_argument('--hidden_feature', type=int, default=64)
        parser.add_argument('--hidden_layers', type=int, default=2)
        parser.add_argument('--num_ensemble', type=int, default=4)
        
        args = parser.parse_args()
        
        return args.__dict__

    def __init__(self, config):
        self.config = config
        setup_seed(self.config['seed'])
        self.env = make_env(config)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.device = self.config['device']

        self.transition = EnsembleTransition(obs_dim=self.state_dim,
                                             action_dim=self.action_dim,
                                             hidden_features=self.config['hidden_feature'],
                                             hidden_layers=self.config['hidden_layers'],
                                             ensemble_size=self.config['num_ensemble'],
                                             with_reward=True)

        actor = CEMActor(self.transition, self.env.action_space, 
                         self.config['horizon'], self.config['samples'],
                         self.config['elites'], self.config['iterations'])

        self.buffer = ray.remote(ReplayBuffer).remote(self.config['buffer_size'])
        self.collector = CollectorServer.remote(self.config, deepcopy(actor), self.buffer, self.config['num_collectors'])
        self.logger = Logger.remote(config, deepcopy(actor), 'pets')

        self.transition = self.transition.to(self.device)
        self.optimizor = torch.optim.Adam(self.transition.parameters(), lr=config['lr'])

    def run(self):
        for i in tqdm(range(self.config['epoch'])):
            batchs_id = self.collector.collect_steps.remote(self.config['data_collection_per_epoch'], self.transition.get_weights())
            batchs = ray.get(batchs_id)
        
            for _ in range(self.config['train_steps_per_epoch']):
                batchs = ray.get(self.buffer.sample.remote(self.config['batch_size']))

                batchs.to_torch(dtype=torch.float32, device=self.device)

                ''' update transition '''
                output_dist = self.transition(batchs['obs'], batchs['action'])
                nll_loss = - output_dist.log_prob(torch.cat([batchs['next_obs'], batchs['reward']], dim=-1)).mean()

                self.optimizor.zero_grad()
                nll_loss.backward()
                self.optimizor.step()

                info = {
                    "nll" : nll_loss.item(),
                }

                self.logger.test_and_log.remote(None, info)

            self.logger.test_and_log.remote(self.transition.get_weights() if i % self.config['test_frequency'] == 0 else None, info)

if __name__ == '__main__':
    ray.init()
    config = PETS.get_config()
    config['seed'] = config['seed'] or random.randint(0, 1000000)
    experiment = PETS(config)
    experiment.run()
