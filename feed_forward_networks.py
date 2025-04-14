"""Training inspired by the training presented at 'Learning to Unknot'"""
from torch import nn

import pfrl

def create_ffn_policy(obs_size: int, action_size: int) -> nn.Module:
    """Create a basic policy network."""
    return nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_size),
        pfrl.policies.SoftmaxCategoricalHead(),
    )

def create_ffn_vf(obs_size: int)-> nn.Module:
    """Create a basic value function network."""
    return nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )


def initialize_ffn(policy, vf):
    for i in range(0, len(policy) - 1, 2):
        if isinstance(policy[i], nn.Linear):
            nn.init.orthogonal_(policy[i].weight, gain=1)
            nn.init.zeros_(policy[i].bias)

    for i in range(0, len(vf), 2):
        if isinstance(vf[i], nn.Linear):
            nn.init.orthogonal_(vf[i].weight, gain=1)
            nn.init.zeros_(vf[i].bias)