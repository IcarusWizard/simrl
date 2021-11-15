import torch
from torch import nn

from simrl.utils.dists import *
from simrl.utils.general import soft_clamp

ACTIVATION_CREATERS = {
    'relu' : lambda dim: nn.ReLU(inplace=True),
    'elu' : lambda dim: nn.ELU(),
    'leakyrelu' : lambda dim: nn.LeakyReLU(negative_slope=0.1, inplace=True),
    'tanh' : lambda dim: nn.Tanh(),
    'sigmoid' : lambda dim: nn.Sigmoid(),
    'identity' : lambda dim: nn.Identity(),
    'prelu' : lambda dim: nn.PReLU(dim),
}

class ShareModule(torch.nn.Module):
    """Module with the ablity to export its weights"""
    def get_weights(self):
        return {k : v.cpu() for k, v in self.state_dict().items()}

class MLP(ShareModule):
    r"""
        Multi-layer Perceptron
        Inputs: 
        
            in_features : int, features numbers of the input
            out_features : int, features numbers of the output
            hidden_features : int, features numbers of the hidden layers
            hidden_layers : int, numbers of the hidden layers
            norm : str, normalization method between hidden layers, default : None
            hidden_activation : str, activation function used in hidden layers, default : 'leakyrelu'
            output_activation : str, activation function used in output layer, default : 'identity'
    """
    def __init__(self, in_features, out_features, hidden_features, hidden_layers, 
                 norm=None, hidden_activation='leakyrelu', output_activation='identity'):
        super(MLP, self).__init__()

        hidden_activation_creater = ACTIVATION_CREATERS[hidden_activation]
        output_activation_creater = ACTIVATION_CREATERS[output_activation]

        if hidden_layers == 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, out_features),
                output_activation_creater(out_features)
            )
        else:
            net = []
            for i in range(hidden_layers):
                net.append(nn.Linear(in_features if i == 0 else hidden_features, hidden_features))
                if norm:
                    if norm == 'ln':
                        net.append(nn.LayerNorm(hidden_features))
                    elif norm == 'bn':
                        net.append(nn.BatchNorm1d(hidden_features))
                    else:
                        raise NotImplementedError(f'{norm} does not supported!')
                net.append(hidden_activation_creater(hidden_features))
            net.append(nn.Linear(hidden_features, out_features))
            net.append(output_activation_creater(out_features))
            self.net = nn.Sequential(*net)

    def forward(self, x):
        r"""forward method of MLP only assume the last dim of x matches `in_features`"""
        head_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        out = self.net(x)
        out = out.view(*head_shape, out.shape[-1])
        return out

class DistributionWrapper(ShareModule):
    r"""wrap output of Module to distribution"""
    BASE_TYPES = ['normal', 'tanhnormal', 'gmm', 'onehot']
    SUPPORTED_TYPES = BASE_TYPES + ['mix']

    def __init__(self, distribution_type='normal', **params):
        super().__init__()
        self.distribution_type = distribution_type
        self.params = params

        assert self.distribution_type in self.SUPPORTED_TYPES, f"{self.distribution_type} is not supported!"

        if self.distribution_type == 'normal':
            self.max_logstd = nn.Parameter(torch.ones(self.params['dim']) * 0, requires_grad=True)
            self.min_logstd = nn.Parameter(torch.ones(self.params['dim']) * -10, requires_grad=True)
            if not self.params.get('conditioned_std', True):
                self.logstd = nn.Parameter(torch.zeros(self.params['dim']), requires_grad=True)
        elif self.distribution_type == 'gmm':
            self.max_logstd = nn.Parameter(torch.ones(self.params['mixture'], self.params['dim']) * 0, requires_grad=True)
            self.min_logstd = nn.Parameter(torch.ones(self.params['mixture'], self.params['dim']) * -10, requires_grad=True)            
            if not self.params.get('conditioned_std', True):
                self.logstd = nn.Parameter(torch.zeros(self.params['mixture'], self.params['dim']), requires_grad=True)
        elif self.distribution_type == 'tanhnormal':
            self.max_logstd = nn.Parameter(torch.ones(self.params['dim']) * 0, requires_grad=True)
            self.min_logstd = nn.Parameter(torch.ones(self.params['dim']) * -10, requires_grad=True)
            if not self.params.get('conditioned_std', True):
                self.logstd = nn.Parameter(torch.zeros(self.params['dim']), requires_grad=True)
            self.register_buffer('loc', torch.as_tensor((params['max'] + params['min']) / 2))
            self.register_buffer('scale', torch.as_tensor((params['max'] - params['min']) / 2))
        elif self.distribution_type == 'mix':
            assert 'dist_config' in self.params.keys(), "You need to provide `dist_config` for Mix distribution"

            self.dist_config = self.params['dist_config']
            self.wrapper_list = []
            self.input_sizes = []
            self.output_sizes = []
            for config in self.dist_config:
                dist_type = config['type']
                assert dist_type in self.SUPPORTED_TYPES, f"{dist_type} is not supported!"
                assert not dist_type == 'mix', "recursive MixDistribution is not supported!"

                self.wrapper_list.append(DistributionWrapper(dist_type, **config))

                self.input_sizes.append(config['dim'])
                self.output_sizes.append(config['output_dim'])
                
            self.wrapper_list = nn.ModuleList(self.wrapper_list)                                     

    def forward(self, x):
        if self.distribution_type == 'normal':
            if self.params.get('conditioned_std', True):
                mu, logstd = torch.chunk(x, 2, dim=-1)
            else:
                mu, logstd = x, self.logstd
            std = torch.exp(soft_clamp(logstd, self.min_logstd, self.max_logstd))
            return DiagnalNormal(mu, std)
        elif self.distribution_type == 'gmm':
            if self.params.get('conditioned_std', True):
                logits, mus, logstds = torch.split(x, [self.params['mixture'], 
                                                       self.params['mixture'] * self.params['dim'], 
                                                       self.params['mixture'] * self.params['dim']], dim=-1)
                mus = mus.view(*mus.shape[:-1], self.params['mixture'], self.params['dim'])      
                logstds = logstds.view(*logstds.shape[:-1], self.params['mixture'], self.params['dim'])
            else:
                logits, mus = torch.split(x, [self.params['mixture'], self.params['mixture'] * self.params['dim']], dim=-1)
                logstds = self.logstd
            stds = torch.exp(soft_clamp(logstds, self.min_logstd, self.max_logstd))
            return GaussianMixture(mus, stds, logits)
        elif self.distribution_type == 'tanhnormal':
            if self.params.get('conditioned_std', True):
                mu, logstd = torch.chunk(x, 2, dim=-1)
            else:
                mu, logstd = x, self.logstd
            mu = 5 * torch.tanh(mu / 5)
            std = torch.exp(soft_clamp(logstd, self.min_logstd, self.max_logstd))
            dist = DiagnalNormal(mu, std)
            transforms = [
                SafeTanhTransform(cache_size=1),
                torch.distributions.transforms.AffineTransform(loc=self.loc, scale=self.scale)
            ]
            return TransformedDistribution(dist, transforms)
        elif self.distribution_type == 'onehot':
            return Onehot(10 * torch.tanh(x / 10)) # stabilize gradients
        elif self.distribution_type == 'mix':
            xs = torch.split(x, self.output_sizes, dim=-1)

            dists = [wrapper(x, _adapt_std, _payload) for x, _adapt_std, _payload, wrapper in zip(xs, self.wrapper_list)]
            
            return MixDistribution(dists)

    def extra_repr(self) -> str:
        return 'type={}, dim={}'.format(
            self.distribution_type, 
            self.params['dim'] if not self.distribution_type == 'mix' else len(self.wrapper_list)
        )

class OnehotActor(ShareModule):
    """Actor in discrete space (output onehot vectors)"""
    def __init__(self, state_dim, action_dim,
                 hidden_features=128,
                 hidden_layers=2,
                 norm=None,
                 hidden_activation='leakyrelu',
                 *args, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.dist_net = torch.nn.Sequential(
            MLP(state_dim, action_dim, hidden_features, hidden_layers, norm, hidden_activation),
            DistributionWrapper(distribution_type='onehot', dim=action_dim)
        )

    def forward(self, state):
        return self.dist_net(state)

    @torch.no_grad()
    def act(self, state, sample_fn=lambda dist: dist.sample()):
        param = next(self.dist_net.parameters())
        device = param.device
        dtype = param.dtype
        state = torch.as_tensor(state, dtype=dtype, device=device).unsqueeze(0)
        action_dist = self.forward(state)
        action = sample_fn(action_dist)
        action = action.squeeze(0).numpy()
        return action

class ContinuousActor(ShareModule):
    """Actor in continuous space"""
    def __init__(self, state_dim, action_dim,
                 hidden_features=128,
                 hidden_layers=2,
                 norm=None,
                 hidden_activation='leakyrelu',
                 conditioned_std=False,
                 *args, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.dist_net = torch.nn.Sequential(
            MLP(state_dim, action_dim * (2 if conditioned_std else 1), hidden_features, hidden_layers, norm, hidden_activation),
            DistributionWrapper(distribution_type='normal', dim=action_dim, conditioned_std=conditioned_std)
        )

    def forward(self, state):
        return self.dist_net(state)

    @torch.no_grad()
    def act(self, state, sample_fn=lambda dist: dist.sample()):
        param = next(self.dist_net.parameters())
        device = param.device
        dtype = param.dtype
        state = torch.as_tensor(state, dtype=dtype, device=device).unsqueeze(0)
        action_dist = self.forward(state)
        action = sample_fn(action_dist)
        action = action.squeeze(0).numpy()
        return action

class BoundedContinuousActor(ShareModule):
    """Actor in continuous space with bound"""
    def __init__(self, state_dim, action_dim,
                 hidden_features=128,
                 hidden_layers=2,
                 norm=None,
                 hidden_activation='leakyrelu',
                 min_action=-1,
                 max_action=1,
                 *args, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.dist_net = torch.nn.Sequential(
            MLP(state_dim, 2 * action_dim, hidden_features, hidden_layers, norm, hidden_activation),
            DistributionWrapper(distribution_type='tanhnormal', dim=action_dim, min=min_action, max=max_action)
        )

    def forward(self, state):
        return self.dist_net(state)

    @torch.no_grad()
    def act(self, state, sample_fn=lambda dist: dist.sample()):
        param = next(self.dist_net.parameters())
        device = param.device
        dtype = param.dtype
        state = torch.as_tensor(state, dtype=dtype, device=device).unsqueeze(0)
        action_dist = self.forward(state)
        action = sample_fn(action_dist)
        action = action.squeeze(0).numpy()
        return action

class Critic(ShareModule):
    """Simple feedforward critic to estimate the value function of state (or state-action pair)"""
    def __init__(self, state_dim, action_dim=None,
                 hidden_features=128,
                 hidden_layers=2,
                 norm=None,
                 hidden_activation='leakyrelu',
                 ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        if self.action_dim: # Q funtion
            self.state_dim += self.action_dim

        self.value_net = MLP(self.state_dim, 1, hidden_features, hidden_layers, norm, hidden_activation)

    def forward(self, state, action=None):
        if self.action_dim:
            assert action is not None
            state = torch.cat([state, action], dim=-1)
        return self.value_net(state)

class DiscreteQ(ShareModule):
    """ Q function for discrete action """
    def __init__(self, state_dim, action_dim,
                 hidden_features=128,
                 hidden_layers=2,
                 norm=None,
                 hidden_activation='leakyrelu',
                 dueling : bool = True,
                 ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling

        if self.dueling:
            self.value_net = MLP(self.state_dim, 1, hidden_features, hidden_layers, norm, hidden_activation)
            self.advantage_net = MLP(self.state_dim, self.action_dim, hidden_features, hidden_layers, norm, hidden_activation)
        else:
            self.q_net = MLP(self.state_dim, self.action_dim, hidden_features, hidden_layers, norm, hidden_activation)

    def forward(self, state : torch.Tensor):
        if self.dueling:
            value = self.value_net(state)
            advantage = self.advantage_net(state)
            advantage = advantage - advantage.mean(dim=-1, keepdim=True)
            return value + advantage
        else:
            return self.q_net(state)

class EnsembleLinear(ShareModule):
    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()

        self.ensemble_size = ensemble_size

        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))

        torch.nn.init.trunc_normal_(self.weight, std=1/(2*in_features**0.5))

        self.register_parameter('saved_weight', torch.nn.Parameter(self.weight.detach().clone()))
        self.register_parameter('saved_bias', torch.nn.Parameter(self.bias.detach().clone()))

        self.select = list(range(0, self.ensemble_size))

    def forward(self, x):
        weight = self.weight[self.select]
        bias = self.bias[self.select]

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias

        return x

    def set_select(self, indexes):
        assert len(indexes) <= self.ensemble_size and max(indexes) < self.ensemble_size
        self.select = indexes
        self.weight.data[indexes] = self.saved_weight.data[indexes]
        self.bias.data[indexes] = self.saved_bias.data[indexes]

    def update_save(self, indexes):
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]

class EnsembleTransition(ShareModule):
    def __init__(self, obs_dim, action_dim, hidden_features, hidden_layers, ensemble_size=7, mode='local', with_reward=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.mode = mode
        self.with_reward = with_reward
        self.ensemble_size = ensemble_size

        self.activation = torch.nn.LeakyReLU(0.1, inplace=True)

        module_list = []
        for i in range(hidden_layers):
            if i == 0:
                module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_features, ensemble_size))
            else:
                module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size))
        self.backbones = torch.nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(hidden_features, 2 * (obs_dim + self.with_reward), ensemble_size)

        self.register_parameter('max_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * 1, requires_grad=True))
        self.register_parameter('min_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * -5, requires_grad=True))

    def forward(self, obs : torch.Tensor, action : torch.Tensor):
        obs_action = torch.cat([obs, action], dim=-1)
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mu, logstd = torch.chunk(self.output_layer(output), 2, dim=-1)
        logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
        if self.mode == 'local':
            if self.with_reward:
                obs, reward = torch.split(mu, [self.obs_dim, 1], dim=-1)
                obs = obs + obs_action[..., :self.obs_dim]
                mu = torch.cat([obs, reward], dim=-1)
            else:
                mu = mu + obs_action[..., :self.obs_dim]
        return DiagnalNormal(mu, torch.exp(logstd))

    def set_select(self, indexes):
        for layer in self.backbones:
            layer.set_select(indexes)
        self.output_layer.set_select(indexes)

    def update_save(self, indexes):
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)