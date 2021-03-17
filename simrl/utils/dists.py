import torch
from torch.functional import F
import numpy as np

EPS = 1e-8

class Normal(torch.distributions.Normal):
    @property
    def mode(self):
        return self.mean

class Onehot:
    """Differentiable Onehot Distribution"""

    def __init__(self, logits):
        """logits -> tensor[*, N]"""
        super().__init__()
        self.n = logits.shape[-1]
        self.logits = logits
        self.p = torch.softmax(logits, dim=-1)
        self.prior = torch.distributions.Categorical(logits=logits)

    def _index2onehot(self, index):
        shape = index.shape
        index = index.view(-1)
        sample = torch.zeros_like(self.logits)
        sample = sample.view(-1, sample.shape[-1])
        sample[np.arange(index.shape[0]), index] = 1
        sample = sample.view(*shape, sample.shape[-1])
        return sample        

    def log_prob(self, x):
        """
        :return: log probability of one hot sample

        """
        index = torch.argmax(x, dim=-1)
        log_prob = self.prior.log_prob(index)
        return log_prob.unsqueeze(-1)
    
    @property
    def mode(self):
        index = torch.argmax(self.logits, dim=-1)
        sample = self._index2onehot(index)
        return sample + self.p - self.p.detach()

    def sample(self, num=torch.Size()):
        # TODO: implement multi-sample mode
        assert len(num) == 0, "currently onehot distribution only support single sample mode"
        index = self.prior.sample(num)
        sample = self._index2onehot(index)
        return sample

    def rsample(self, num=torch.Size()):
        # Implement straight-through estimator
        # Bengio et.al. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation 
        sample = self.sample(num)
        return sample + self.p - self.p.detach()

    def sample_with_logprob(self, num=torch.Size()):
        x = self.rsample(num)
        return x, self.log_prob(x)
    
    def entropy(self):
        entropy = self.prior.entropy()
        return entropy.unsqueeze(-1)

class GaussianMixture:

    def __init__(self, logits, mus, stds):
        """
            logits  ->  tensor[*, M]
            mus     ->  tensor[*, M, N]
            stds    ->  tensor[*, M, N]
            * can be any shape, M is number of mixture, N is dim of each gauss
        """
        super().__init__()
        self.prior = Onehot(logits)
        self.mus = mus
        self.stds = stds
        self.dists = [Normal(mus.select(-2, i), stds.select(-2, i)) for i in range(self.prior.n)]

    def log_prob(self, x):
        log_probs = torch.stack([dist.log_prob(x) for dist in self.dists], dim=-2)
        return torch.log((self.prior.p.unsqueeze(-1) * torch.exp(log_probs)).sum(-2))
    
    @property
    def mode(self):
        raise NotImplementedError()

    def sample(self, num=torch.Size()):
        # TODO: implement multi-sample mode
        assert len(num) == 0, "currently GaussianMixture distribution only support single sample mode"
        onehot = self.prior.sample(num) # [*, M]
        samples = torch.stack([dist.sample(num) for dist in self.dists], dim=-2) # [*, M, N]
        sample = (samples * onehot.unsqueeze(-1)).sum(-2)
        return sample

    def rsample(self, num=torch.Size()):
        # TODO: implement multi-sample mode
        assert len(num) == 0, "currently GaussianMixture distribution only support single sample mode"
        onehot = self.prior.rsample(num) # [*, M]
        samples = torch.stack([dist.rsample(num) for dist in self.dists], dim=-2) # [*, M, N]
        sample = (samples * onehot.unsqueeze(-1)).sum(-2)
        return sample

    def sample_with_logprob(self, num=torch.Size()):
        x = self.rsample(num)
        return x, self.log_prob(x)
    
    def entropy(self):
        entropys = torch.stack([dist.entropy() for dist in self.dists], dim=-2)
        return (entropys * self.prior.p.unsqueeze(-1)).sum(-2)

class MixDistribution:
    """Collection of different distributions"""
    
    def __init__(self, dists):
        super().__init__()
        self.dists = dists

        self.sizes = []
        for dist in self.dists:
            if type(dist) == Normal:
                self.sizes.append(dist.mean.shape[-1])
            elif type(dist) == Onehot:
                self.sizes.append(dist.n)
            elif type(dist) == GaussianMixture:
                self.sizes.append(dist.mus.shape[-1])
            else:
                raise NotImplementedError(f"distribution type {type(dist)} is not support!")

    def log_prob(self, x):
        if type(x) == list:
            return [self.dists[i].log_prob(x[i]) for i in range(len(x))]
        # manually split the tensor
        x = torch.split(x, self.sizes, dim=-1)
        log_probs = [self.dists[i].log_prob(x[i]) for i in range(len(x))]
        return torch.cat(log_probs, dim=-1)

    @property
    def mode(self):
        def find_mode(dist):
            if type(dist) == Normal:
                return dist.mean
            elif type(dist) == Onehot:
                return dist.mode 
            else:
                raise NotImplementedError(f"distribution type {type(dist)} is not support!")         
        modes = list(map(find_mode, self.dists))
        return torch.cat(modes, dim=-1)

    def sample(self, num=torch.Size()):
        samples = [dist.sample(num) for dist in self.dists]
        return torch.cat(samples, dim=-1)

    def rsample(self, num=torch.Size()): 
        samples = [dist.rsample(num) for dist in self.dists]
        return torch.cat(samples, dim=-1)

    def sample_with_logprob(self, num=torch.Size()):
        x = self.rsample(num)
        return x, self.log_prob(x)
    
    def entropy(self):
        entropys = [dist.entropy() for dist in self.dists]
        return torch.cat(entropys, dim=-1)