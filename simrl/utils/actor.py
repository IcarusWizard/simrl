import torch
import random
import numpy as np
from abc import ABC
from typing import Dict, Optional

from simrl.utils.modules import DiscreteQ

class Actor(ABC):
    ''' Actor class is used to interact with the environment '''

    @torch.no_grad()
    def act(self, state : np.ndarray, *args, **kwargs) -> np.ndarray:
        ''' take action on the given state '''
        raise NotImplementedError

    def reset(self) -> None:
        ''' reset the actor to recieve new trajectory, useful when the actor is stateless '''
        pass

    def set_parameters(self, parameters : Dict[str, torch.Tensor]) -> None:
        ''' set parameters for networks ''' 
        raise NotImplementedError

class DistributionActor(Actor):
    ''' Actor for policy with distributional interface '''
    def __init__(self, model) -> None:  
        self.model = model

    @torch.no_grad()
    def act(self, state: np.ndarray, sample_fn=lambda dist: dist.sample(), *args, **kwargs) -> np.ndarray:
        param = next(self.model.parameters())
        device = param.device
        dtype = param.dtype
        state = torch.as_tensor(state, dtype=dtype, device=device).unsqueeze(0)
        action_dist = self.model(state)
        action = sample_fn(action_dist)
        action = action.squeeze(0).numpy()
        return action 

    def set_parameters(self, parameters : Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(parameters)

class EpsilonGreedyActor(Actor):
    ''' Actor for q-learning policy '''
    def __init__(self, q_func : DiscreteQ, epsilon : float) -> None:
        self.q_func = q_func
        self.action_dim = self.q_func.action_dim
        self.epsilon = epsilon

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon : Optional[float]=None, *args, **kwargs) -> np.ndarray:
        epsilon = epsilon or self.epsilon

        if random.random() < epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            param = next(self.q_func.parameters())
            device = param.device
            dtype = param.dtype
            state = torch.as_tensor(state, dtype=dtype, device=device).unsqueeze(0)
            q_value = self.q_func(state)
            action = torch.argmax(q_value).item()
        
        onehot_action = np.zeros(self.action_dim)
        onehot_action[action] = 1.0

        return onehot_action

    def set_parameters(self, parameters : Dict[str, torch.Tensor]) -> None:
        self.q_func.load_state_dict(parameters)

class RandomShootingActor(Actor):
    ''' Actor for random shooting '''
    def __init__(self, transition, action_space, horizon, samples) -> None:
        self.transition = transition
        self.action_space = action_space
        self.horizon = horizon
        self.samples = samples

    @torch.no_grad()
    def act(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        param = next(self.transition.parameters())
        device = param.device
        dtype = param.dtype
        state = torch.as_tensor(state, dtype=dtype, device=device).unsqueeze(0).repeat(self.samples, 1)
        actions = torch.as_tensor(np.stack([self.action_space.sample() for _ in range(self.horizon * self.samples)]), dtype=dtype, device=device)
        actions = actions.view(self.horizon, self.samples, actions.shape[-1])

        total_reward = 0
        for t in range(self.horizon):
            output_dist = self.transition(state, actions[t])
            output = output_dist.mode.mean(dim=0)
            state = output[..., :-1]
            total_reward += output[..., -1]

        max_index = torch.argmax(total_reward)
        action = actions[0, max_index]
        action = action.numpy()

        return action 

    def set_parameters(self, parameters : Dict[str, torch.Tensor]) -> None:
        self.transition.load_state_dict(parameters)