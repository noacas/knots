"""Training inspired by the training presented at 'Learning to Unknot'"""
from torch import nn

import pfrl

def policy(obs_size, action_size):
    """Create a basic policy network."""
    return nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_size),
        pfrl.policies.SoftmaxCategoricalHead(),
    )

def value_function(obs_size):
    """Create a basic value function network."""
    return nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )
