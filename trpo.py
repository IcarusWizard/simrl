import ray
import copy
from tianshou.data import batch
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm

from simrl.utils import setup_seed, compute_gae
from simrl.utils.modules import OnehotActor, ContinuousActor, Critic
from simrl.utils.envs import make_env
from simrl.utils.data import CollectorServer, ReplayBuffer
from simrl.utils.logger import Logger
from simrl.utils.dists import kl_divergence

def cg(Ax, b, cg_iters : int = 10):
    """
    Conjugate gradient algorithm
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
    """
    x = torch.zeros_like(b)
    r = b.clone() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    p = r.clone()
    r_dot_old = torch.dot(r, r)
    for _ in range(cg_iters):
        z = Ax(p)
        with torch.no_grad():
            alpha = r_dot_old / (torch.dot(p, z) + 1e-8)
            x = x + alpha * p
            r = r - alpha * z
            r_dot_new = torch.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
    return x

class TRPO:
    @staticmethod
    def get_config():
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='CartPole-v1')
        parser.add_argument('--seed', type=int, default=None)
        parser.add_argument('--v_lr', type=float, default=1e-3)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--lambda', type=float, default=0.97)
        parser.add_argument('--delta', type=float, default=0.01)
        parser.add_argument('--epoch', type=int, default=100)
        parser.add_argument('--step_per_epoch', type=int, default=4000)
        parser.add_argument('--num_collectors', type=int, default=4)
        parser.add_argument('--train_v_iters', type=int, default=80)
        parser.add_argument('--cg_iters', type=int, default=10)
        parser.add_argument('--damping_coeff', type=float, default=0.1)
        parser.add_argument('--backtrack_iters', type=int, default=10)
        parser.add_argument('--backtrack_coeff', type=float, default=0.8)
        parser.add_argument('--test-num', type=int, default=20)
        parser.add_argument('--test_frequency', type=int, default=5)
        parser.add_argument('--log_video', action='store_true')
        parser.add_argument('--log', type=str, default=None)
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

        parser.add_argument('-ahf', '--actor_hidden_features', type=int, default=64)
        parser.add_argument('-ahl', '--actor_hidden_layers', type=int, default=2)
        parser.add_argument('-aa', '--actor_activation', type=str, default='tanh')
        parser.add_argument('-an', '--actor_norm', type=str, default=None)
        parser.add_argument('-chf', '--critic_hidden_features', type=int, default=64)
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
                                 norm=self.config['actor_norm'])

        self.critic = Critic(self.state_dim,
                             hidden_features=self.config['critic_hidden_features'],
                             hidden_layers=self.config['critic_hidden_layers'],
                             hidden_activation=self.config['critic_activation'],
                             norm=self.config['critic_norm'])

        self.buffer = ray.remote(ReplayBuffer).remote(100000)
        self.collector = CollectorServer.remote(self.config, copy.deepcopy(self.actor), self.buffer, self.config['num_collectors'])
        self.logger = Logger.remote(config, copy.deepcopy(self.actor), 'trpo')

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        self.critic_optimizor = torch.optim.Adam(self.critic.parameters(), lr=config['v_lr'])

    def run(self):
        for i in tqdm(range(self.config['epoch'])):
            ray.get(self.collector.collect_steps.remote(self.config["step_per_epoch"], self.actor.get_weights()))

            batchs = ray.get(self.buffer.pop.remote())
            batchs.to_torch(dtype=torch.float32, device=self.device)

            ''' process data '''
            with torch.no_grad():
                # normalize rewards
                reward_std = batchs.reward.std()
                if not reward_std == 0:
                    batchs.reward = (batchs.reward - batchs.reward.mean()) / reward_std
            
                old_action_dist = self.actor(batchs.obs)
                batchs.log_prob = old_action_dist.log_prob(batchs.action).detach()
                batchs.value = self.critic(batchs.obs).detach()
                batchs.adv, batchs.ret = compute_gae(batchs.reward, batchs.value, batchs.done, 
                                                     self.config['gamma'], self.config['lambda'])

            ''' update actor '''
            action_dist = self.actor(batchs.obs)
            log_prob = action_dist.log_prob(batchs.action)
            ratio = (log_prob - batchs.log_prob).exp().unsqueeze(dim=-1)

            p_loss = - (ratio * batchs.adv).mean()

            p_grad = torch.cat([grad.view(-1) for grad in torch.autograd.grad(p_loss, self.actor.parameters(), create_graph=True)])
            kl = kl_divergence(action_dist, old_action_dist).mean()
            kl_grad = torch.cat([grad.view(-1) for grad in torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)])
            def hvp(x):
                return torch.cat([grad.view(-1) for grad in torch.autograd.grad(torch.sum(kl_grad * x), self.actor.parameters(), create_graph=True)]) + \
                    self.config['damping_coeff'] * x

            x = cg(hvp, p_grad, cg_iters=self.config['cg_iters'])
            total_grad = torch.sqrt(2 * self.config['delta'] / (torch.dot(x, hvp(x)) + 1e-8)) * x

            old_parameters = torch.cat([param.view(-1) for param in self.actor.parameters()])
            @torch.no_grad()
            def set_and_eval(new_paramters):
                parameter_index = 0
                for parameter in self.actor.parameters():
                    total_shape = np.prod(parameter.shape)
                    _param = new_paramters[parameter_index:parameter_index+total_shape].view(parameter.shape)
                    parameter_index += total_shape
                    parameter.data = _param
                new_action_dist = self.actor(batchs.obs)
                kl = kl_divergence(new_action_dist, old_action_dist).mean()
                return kl.item()

            for j in range(self.config['backtrack_iters']):
                alpha = self.config['backtrack_coeff'] ** j
                new_parameters = old_parameters - alpha * total_grad
                new_kl = set_and_eval(new_parameters)
                if new_kl < self.config['delta']: break

            ''' update critic '''
            for _ in range(self.config['train_v_iters']):
                value = self.critic(batchs.obs)
                v_loss = torch.mean((value - batchs.ret) ** 2)

                self.critic_optimizor.zero_grad()
                v_loss.backward()
                self.critic_optimizor.step()

            info = {
                "ratio" : ratio.mean().item(),
                "p_loss" : p_loss.item(),
                "v_loss" : v_loss.item(),
                "alpha" : alpha,
                "kl" : new_kl,
            }

            self.logger.test_and_log.remote(self.actor.get_weights() if i % self.config['test_frequency'] == 0 else None, info)

if __name__ == '__main__':
    ray.init()
    config = TRPO.get_config()
    config['seed'] = config['seed'] or random.randint(0, 1000000)
    experiment = TRPO(config)
    experiment.run()
