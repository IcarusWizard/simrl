import gym
import torch
import random
import numpy as np
from abc import ABC
from copy import deepcopy
from typing import Dict, Optional

from simrl.utils.modules import DiscreteQ
from simrl.utils.dists import DiagnalNormal, Onehot

class Actor(ABC):
    ''' Actor class is used to interact with the environment '''

    @torch.no_grad()
    def act(self, state : np.ndarray, *args, **kwargs) -> np.ndarray:
        ''' take action on the given state '''
        raise NotImplementedError

    def reset(self) -> None:
        ''' reset the actor to recieve new trajectory, useful when the actor is stateful '''
        pass

    def set_parameters(self, parameters : Dict[str, torch.Tensor]) -> None:
        ''' set parameters for networks ''' 
        raise NotImplementedError

    def to(self, device : str):
        ''' change device if possible '''
        pass

class RandomActor(Actor):
    ''' Actor takes actions sampled from uniform distribution, useful to fill initial buffer '''

    def __init__(self, action_space : gym.Space) -> None:
        super().__init__()
        self.action_space = action_space

    def act(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self.action_space.sample()

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        pass

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

    def to(self, device : str):
        self.model = self.model.to(device)

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

    def to(self, device : str):
        self.q_func = self.q_func.to(device)

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
            output = output_dist.sample()
            model_index = np.random.randint(0, output.shape[0], self.samples)
            output = output[model_index, np.arange(self.samples)]
            state = output[..., :-1]
            total_reward += output[..., -1]

        max_index = torch.argmax(total_reward)
        action = actions[0, max_index]
        action = action.cpu().numpy()

        return action 

    def set_parameters(self, parameters : Dict[str, torch.Tensor]) -> None:
        self.transition.load_state_dict(parameters)

    def to(self, device : str):
        self.transition = self.transition.to(device)


class CEMActor(Actor):
    ''' Actor for random shooting with CEM iterations '''
    def __init__(self, transition, action_space, horizon, samples, elites, iterations, save_plan=True) -> None:
        self.transition = transition
        self.action_space = action_space
        self.continuous = isinstance(self.action_space, gym.spaces.Box)
        self.horizon = horizon
        self.samples = samples
        self.elites = elites
        self.iterations = iterations
        self.save_plan = save_plan
        self.plan_dist = None

    def reset(self) -> None:
        self.plan_dist = None

    @torch.no_grad()
    def act(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        param = next(self.transition.parameters())
        device = param.device
        dtype = param.dtype
        init_state = torch.as_tensor(state, dtype=dtype, device=device).unsqueeze(0).repeat(self.samples, 1)

        for i in range(self.iterations):
            state = deepcopy(init_state)
            if i == 0: 
                if self.plan_dist is None: # initialize with uniform samples
                    actions = torch.as_tensor(np.stack([self.action_space.sample() for _ in range(self.horizon * self.samples)]), dtype=dtype, device=device)
                    actions = actions.view(self.horizon, self.samples, actions.shape[-1])
                else:
                    last_action = torch.as_tensor(np.stack([self.action_space.sample() for _ in range(self.samples)]), dtype=dtype, device=device).unsqueeze(dim=0)
                    actions = torch.cat([self.plan_dist.sample((self.samples,)).permute(1, 0, 2).contiguous(), last_action], dim=0)
            else:
                actions = action_dist.sample((self.samples,)).permute(1, 0, 2).contiguous()

            total_reward = 0
            for t in range(self.horizon):
                output_dist = self.transition(state, actions[t])
                output = output_dist.sample()
                model_index = np.random.randint(0, output.shape[0], self.samples)
                output = output[model_index, np.arange(self.samples)]
                state = output[..., :-1]
                total_reward += output[..., -1]

            elite_index = torch.argsort(total_reward)[-self.elites:]
            actions = actions[:, elite_index]

            if self.continuous:
                action_dist = DiagnalNormal(actions.mean(dim=1), actions.std(dim=1))
            else:
                action_dist = Onehot(probs=actions.mean(dim=1))

        if self.save_plan:
            if self.continuous:
                self.plan_dist = DiagnalNormal(action_dist.mode[1:], action_dist.std[1:] + 1e-4)
            else:
                self.plan_dist = Onehot(probs=action_dist.probs[1:])

        action = actions[0, -1]
        action = action.cpu().numpy()

        return action 

    def set_parameters(self, parameters : Dict[str, torch.Tensor]) -> None:
        self.transition.load_state_dict(parameters)

    def to(self, device : str):
        self.transition = self.transition.to(device)