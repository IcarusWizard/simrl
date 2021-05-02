import ray
import torch
import random
import argparse
from tqdm import tqdm
from copy import deepcopy

from simrl.utils import setup_seed, soft_critic_update
from simrl.utils.modules import DiscreteQ
from simrl.utils.envs import make_env
from simrl.utils.data import Collector, ReplayBuffer
from simrl.utils.logger import Logger
from simrl.utils.actor import EpsilonGreedyActor

class DQN:
    @staticmethod
    def get_config():
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='CartPole-v1')
        parser.add_argument('--seed', type=int, default=None)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--buffer_size', type=int, default=int(1e6))
        parser.add_argument('--epsilon', type=float, default=0.1)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--epoch', type=int, default=1000)
        parser.add_argument('--update_period', type=int, default=100)
        parser.add_argument('--num_collectors', type=int, default=4)
        parser.add_argument('--training_step_per_epoch', type=int, default=1000)
        parser.add_argument('--test-num', type=int, default=20)
        parser.add_argument('--test_frequency', type=int, default=5)
        parser.add_argument('--log_video', action='store_true')
        parser.add_argument('--log', type=str, default=None)
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

        parser.add_argument('-ahf', '--q_hidden_features', type=int, default=128)
        parser.add_argument('-ahl', '--q_hidden_layers', type=int, default=2)
        parser.add_argument('-aa', '--q_activation', type=str, default='leakyrelu')
        parser.add_argument('-an', '--q_norm', type=str, default=None)
        parser.add_argument('--dueling', type=lambda x: [False, True][int(x)], default=True)
        
        args = parser.parse_args()
        
        return args.__dict__

    def __init__(self, config):
        self.config = config
        setup_seed(self.config['seed'])
        self.env = make_env(config)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.device = self.config['device']

        assert self.config['env_type'] == 'discrete', 'DQN can only handle discrete environment!'

        self.q_func1 = DiscreteQ(self.state_dim, self.action_dim, 
                                hidden_features=self.config['q_hidden_features'],
                                hidden_layers=self.config['q_hidden_layers'],
                                norm=self.config['q_norm'],
                                hidden_activation=self.config['q_activation'],
                                dueling=self.config['dueling'])

        self.q_func2 = DiscreteQ(self.state_dim, self.action_dim, 
                                hidden_features=self.config['q_hidden_features'],
                                hidden_layers=self.config['q_hidden_layers'],
                                norm=self.config['q_norm'],
                                hidden_activation=self.config['q_activation'],
                                dueling=self.config['dueling'])

        self.buffer = ray.remote(ReplayBuffer).remote(self.config['buffer_size'])
        self.collector = Collector.remote(self.config, deepcopy(EpsilonGreedyActor(self.q_func1, epsilon=self.config['epsilon'])))
        self.logger = Logger.remote(config, deepcopy(EpsilonGreedyActor(self.q_func1, epsilon=0.0)), 'dqn')

        self.q_func1 = self.q_func1.to(self.device)
        self.q_func2 = self.q_func2.to(self.device)
        self.q_target1 = deepcopy(self.q_func1)
        self.q_target2 = deepcopy(self.q_func2)

        self.optimizor = torch.optim.Adam([*self.q_func1.parameters(), *self.q_func2.parameters()], lr=config['lr'])

    def run(self):
        batchs = ray.get(self.collector.collect_steps.remote(1000, self.q_func1.get_weights()))
        ray.get(self.buffer.put.remote(batchs))

        step = 0
        for i in tqdm(range(self.config['epoch'])):
            for j in range(self.config['training_step_per_epoch']):
                step += 1
                batchs = ray.get(self.collector.collect_steps.remote(1, self.q_func1.get_weights()))
                ray.get(self.buffer.put.remote(batchs))
        
                batchs = ray.get(self.buffer.sample.remote(self.config['batch_size']))
                batchs.to_torch(dtype=torch.float32, device=self.device)

                ''' update q '''
                with torch.no_grad():
                    next_q = torch.min(
                        self.q_target1(batchs['next_obs']).max(dim=-1, keepdim=True)[0],
                        self.q_target2(batchs['next_obs']).max(dim=-1, keepdim=True)[0],
                    )
                    target = batchs['reward'] + self.config['gamma'] * (1 - batchs['done']) * next_q

                current_q1 = torch.sum(self.q_func1(batchs['obs']) * batchs['action'], dim=-1, keepdim=True)
                current_q2 = torch.sum(self.q_func2(batchs['obs']) * batchs['action'], dim=-1, keepdim=True)
                q_loss = torch.mean((current_q1 - target) ** 2) + torch.mean((current_q2 - target) ** 2)

                self.optimizor.zero_grad()
                q_loss.backward()
                self.optimizor.step()

                if step % self.config['update_period'] == 0:
                    soft_critic_update(self.q_func1, self.q_target1, 1.0)
                    soft_critic_update(self.q_func2, self.q_target2, 1.0)

                info = {
                    "q_loss" : q_loss.item(),
                }

                self.logger.test_and_log.remote(None, info)

            self.logger.test_and_log.remote(self.q_func1.get_weights() if i % self.config['test_frequency'] == 0 else None, info)

if __name__ == '__main__':
    ray.init()
    config = DQN.get_config()
    config['seed'] = config['seed'] or random.randint(0, 1000000)
    experiment = DQN(config)
    experiment.run()
