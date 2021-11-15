import ray
import copy
import torch
import random
import argparse
from tqdm import tqdm

from simrl.utils import setup_seed, compute_gae
from simrl.utils.modules import OnehotActor, ContinuousActor, Critic
from simrl.utils.envs import make_env
from simrl.utils.data import CollectorServer, ReplayBuffer
from simrl.utils.logger import Logger
from simrl.utils.actor import DistributionActor

class PPO:

    @staticmethod
    def get_config():
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='CartPole-v1')
        parser.add_argument('--seed', type=int, default=None, help='random seed used in the run, random generate the seed if not specified')
        parser.add_argument('--lr', type=float, default=3e-4, help='learning rate for both actor and critic')
        parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for value function')
        parser.add_argument('--lambda', type=float, default=0.95, help='lambda for gae')
        parser.add_argument('--epsilon', type=float, default=0.2, help='bound of the important ratio')
        parser.add_argument('--entropy_coef', type=float, default=0, help='entropy loss scale')
        parser.add_argument('--grad_clip', type=float, default=0.5, help='max value of the gradient norm')
        parser.add_argument('--epoch', type=int, default=1000, help='total epoch of training')
        parser.add_argument('--data_collection_per_epoch', type=int, default=2048)
        parser.add_argument('--num_collectors', type=int, default=4, help='number of ray actors to collect data')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size of each policy update')
        parser.add_argument('--ppo_run', type=int, default=4, help='number of iteration over the collected data')
        parser.add_argument('--test-num', type=int, default=10, help='number of episode of online test')
        parser.add_argument('--test_frequency', type=int, default=10, help='how many epoch to perform an online test')
        parser.add_argument('--log_video', action='store_true', help='whether to store a video of online test')
        parser.add_argument('--log', type=str, default=None, help='log dir, use the random seed if not specified')
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

        parser.add_argument('-ahf', '--actor_hidden_features', type=int, default=64)
        parser.add_argument('-ahl', '--actor_hidden_layers', type=int, default=2)
        parser.add_argument('-aa', '--actor_activation', type=str, default='tanh')
        parser.add_argument('-an', '--actor_norm', type=str, default=None)
        parser.add_argument('-cs', '--conditioned_std', action='store_true')
        parser.add_argument('-chf', '--critic_hidden_features', type=int, default=64)
        parser.add_argument('-chl', '--critic_hidden_layers', type=int, default=2)
        parser.add_argument('-ca', '--critic_activation', type=str, default='tanh')
        parser.add_argument('-cn', '--critic_norm', type=str, default=None)
        
        args = parser.parse_known_args()[0]
        
        return args.__dict__

    def __init__(self, config):
        self.config = config
        setup_seed(self.config['seed'])
        self.env = make_env(config)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.device = torch.device(self.config['device'])

        if self.config['env_type'] == 'discrete':
            actor_class = OnehotActor
        elif self.config['env_type'] == 'continuous':
            actor_class = ContinuousActor
        else:
            raise ValueError('{} is not supported!'.format(self.config['env_type']))
        
        self.actor = actor_class(self.state_dim, self.action_dim,
                                 hidden_features=self.config['actor_hidden_features'],
                                 hidden_layers=self.config['actor_hidden_layers'],
                                 hidden_activation=self.config['actor_activation'],
                                 norm=self.config['actor_norm'],
                                 conditioned_std=config['conditioned_std'])

        self.critic = Critic(self.state_dim,
                             hidden_features=self.config['critic_hidden_features'],
                             hidden_layers=self.config['critic_hidden_layers'],
                             hidden_activation=self.config['critic_activation'],
                             norm=self.config['critic_norm'])

        self.buffer = ray.remote(ReplayBuffer).remote(100000)
        self.collector = CollectorServer.remote(self.config, copy.deepcopy(DistributionActor(self.actor)), self.buffer, self.config['num_collectors'])
        self.logger = Logger.remote(config, copy.deepcopy(DistributionActor(self.actor)), 'ppo')

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        self.optimizor = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=config['lr'])

    def run(self):
        for i in tqdm(range(self.config['epoch'])):
            ray.get(self.collector.collect_steps.remote(self.config["data_collection_per_epoch"], self.actor.get_weights()))

            batchs = ray.get(self.buffer.pop.remote())
            batchs.to_torch(dtype=torch.float32, device=self.device)

            # normalize rewards
            reward_std = batchs.reward.std()
            if not reward_std == 0: # reward all the same, meaning it is a survival reward
                batchs.reward = (batchs.reward - batchs.reward.mean()) / (reward_std + 1e-4)

            for _ in range(self.config['ppo_run']):
                # recompute the advantage every iteration
                action_dist = self.actor(batchs.obs)
                batchs.log_prob = action_dist.log_prob(batchs.action).detach()
                batchs.value = self.critic(batchs.obs).detach()
                batchs.adv, batchs.ret = compute_gae(batchs.reward, batchs.value, batchs.done, 
                                                     self.config['gamma'], self.config['lambda'])

                for batch in batchs.split(self.config['batch_size']):
                    action_dist = self.actor(batch.obs)
                    log_prob = action_dist.log_prob(batch.action)
                    ratio = (log_prob - batch.log_prob).exp().unsqueeze(dim=-1) # important ratio

                    surr1 = ratio * batch.adv
                    surr2 = torch.clamp(ratio, 1 - self.config['epsilon'], 1 + self.config['epsilon']) * batch.adv
                    p_loss = - torch.min(surr1, surr2).mean()

                    value = self.critic(batch.obs)
                    v_loss = torch.mean((value - batch.ret) ** 2)

                    e_loss = - action_dist.entropy().mean()

                    loss = p_loss + v_loss + e_loss * self.config['entropy_coef']

                    self.optimizor.zero_grad()
                    loss.backward()  
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['grad_clip'])       
                    self.optimizor.step()    

                    info = {
                        "ratio" : ratio.mean().item(),
                        "p_loss" : p_loss.item(),
                        "v_loss" : v_loss.item(),
                        "e_loss" : e_loss.item(),
                        "grad_norm" : grad_norm.item(),
                    }

            logger_task_id = self.logger.test_and_log.remote(self.actor.get_weights() if i % self.config['test_frequency'] == 0 else None, info)

        ray.get(logger_task_id) # make sure the log is complete before exit

if __name__ == '__main__':
    ray.init()
    config = PPO.get_config()
    config['seed'] = config['seed'] or random.randint(0, 1000000)
    experiment = PPO(config)
    experiment.run()
