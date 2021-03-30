import torch
import random
import numpy as np
from torch.functional import F

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def compute_gae(reward, value, done, gamma=0.99, lam=0.95):

    returns = torch.zeros_like(reward)
    advantages = torch.zeros_like(reward)

    td_error = torch.zeros_like(reward)
    pre_value, pre_adv = 0, 0
    for t in reversed(range(reward.shape[0])):
        td_error[t] = reward[t] + gamma * pre_value * (1 - done[t]) - value[t]
        advantages[t] = td_error[t] + gamma * lam * pre_adv * (1 - done[t])
        pre_adv = advantages[t]
        pre_value = value[t]
    returns = value + advantages
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

    return advantages, returns

def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

@torch.no_grad()
def soft_critic_update(source, target, tau=0.005):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data = (1 - tau) * target_param.data + tau * source_param.data

@torch.no_grad()
def hard_critic_update(source, target):
    target.load_state_dict(source.state_dict())