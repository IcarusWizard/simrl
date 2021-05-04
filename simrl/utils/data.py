import ray
import torch
import numpy as np
from copy import deepcopy
from tianshou.data import Batch
from typing import Dict, Any

from .envs import make_env
from .actor import Actor

@ray.remote
class Collector:
    def __init__(self, config : Dict[str, Any], actor : Actor):
        self.config = config
        self.env = make_env(self.config)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.actor = actor
        self.o = self.env.reset()
        self.actor.reset()

    def collect_steps(self, steps : int, parameters : Dict[str, torch.Tensor]):
        self.actor.set_parameters(parameters)

        batchs = []

        for _ in range(steps):
            a = self.actor.act(self.o)
            n_o, r, d, info = self.env.step(a)

            batchs.append(Batch(
                obs=self.o,
                next_obs=n_o,
                reward=[r],
                action=a,
                done=[float(d)],
            ))

            if d:
                self.o = self.env.reset()
                self.actor.reset()
            else:
                self.o = n_o

        batchs = Batch.stack(batchs)

        return batchs

    def collect_trajectory(self, parameters : Dict[str, torch.Tensor]):
        self.actor.set_parameters(parameters)

        batchs = []
        self.o = self.env.reset()
        self.actor.reset()
        while True:
            a = self.actor.act(self.o)
            n_o, r, d, info = self.env.step(a)

            batchs.append(Batch(
                obs=self.o,
                next_obs=n_o,
                reward=[r],
                action=a,
                done=[float(d)],
            ))

            if d:
                self.o = self.env.reset()
                self.actor.reset()
                break
            else:
                self.o = n_o

        batchs = Batch.stack(batchs)

        return batchs

    def set_actor(self, actor : Actor):
        self.actor = actor
        self.actor.reset()

    def print_env_info(self):
        print('env name:', self.config['env'])
        print('env type:', self.config['env_type'])
        print('state dim:', self.state_dim)
        print('action dim:', self.action_dim)
        if self.config['env_type'] == 'continuous':
            print('min action:', self.env.action_space.low)
            print('max action:', self.env.action_space.high)
        
@ray.remote
class CollectorServer:
    def __init__(self, config, actor, buffer, num_collectors):
        self.config = config
        self.buffer = buffer
        self.num_collectors = num_collectors
        self.collectors = [Collector.remote(config, actor) for _ in range(num_collectors)]
        ray.get(self.collectors[0].print_env_info.remote())

    def collect_steps(self, steps, actor_state_dict):
        collected_steps = 0

        collection_jobs = [self.collectors[i].collect_trajectory.remote(actor_state_dict) for i in range(self.num_collectors)]
        while collected_steps < steps:
            ready, _ = ray.wait(collection_jobs)
            ready_id = ready[0]

            index = collection_jobs.index(ready_id)
            result = ray.get(ready_id)
            collection_jobs[index] = self.collectors[index].collect_trajectory.remote(actor_state_dict)
            
            self.buffer.put.remote(result)
            collected_steps += result.shape[0]

        return collected_steps

    def set_actor(self, actor : Actor):
        ray.get([collector.set_actor.remote(deepcopy(actor)) for collector in self.collectors])


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.
    """
    def __init__(self, buffer_size : int):
        self.data = None
        self.buffer_size = int(buffer_size)

    def put(self, batch_data : Batch):
        batch_data.to_torch(device='cpu')

        if self.data is None:
            self.data = batch_data
        else:
            self.data.cat_(batch_data)
        
        if len(self) > self.buffer_size:
            self.data = self.data[len(self) - self.buffer_size : ]

    def __len__(self):
        if self.data is None: return 0
        return self.data.shape[0]

    def sample(self, batch_size : int):
        assert len(self) > 0, 'Cannot sample from an empty buffer!'
        indexes = np.random.randint(0, len(self), size=(batch_size))
        return self.data[indexes]

    def pop(self):
        ''' Pop up all the data '''
        data = self.data
        self.data = None
        return data