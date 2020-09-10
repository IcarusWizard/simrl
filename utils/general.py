import torch
import random
import numpy as np

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
    returns = (returns - returns.mean()) / (returns.std() + 1e-4)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

    return advantages, returns