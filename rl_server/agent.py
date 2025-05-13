from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from godot_rl.wrappers.clean_rl_wrapper import CleanRLGodotEnv
from torch.distributions.normal import Normal

_ACT: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU}


class FeedForward(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_total_layers: int,
        activator: type[nn.Module],
    ):
        layers: list[nn.Module] = [nn.Flatten()]
        for i in range(num_total_layers):
            in_feat = in_features if i == 0 else hidden_features
            out_feat = out_features if i == num_total_layers - 1 else hidden_features
            layers.append(nn.Linear(in_feat, out_feat))
            if i != num_total_layers - 1:
                layers.append(activator())
        super().__init__(*layers)


class Agent(nn.Module):
    def __init__(
        self,
        envs: CleanRLGodotEnv,
        num_layers: int = 1,
        inter_factor: float = 1,
        activation: str = "tanh",
    ):
        """
        Args:
            num_layers: Total number of layers
            inter_factor: Factor of intermediate dimensions from input features
            activation: Type of activation layer
        """

        super().__init__()
        in_features = np.prod(envs.single_observation_space.shape).item()
        assert envs.single_action_space.shape is not None
        out_features = np.prod(envs.single_action_space.shape).item()

        assert num_layers >= 0
        assert activation in _ACT, f"activation must be one of {list(_ACT.keys())}"

        hidden_dim = int(in_features * inter_factor)
        self.critic = FeedForward(
            in_features, hidden_dim, 1, num_layers, _ACT[activation]
        )
        self.actor_mean = FeedForward(
            in_features, hidden_dim, out_features, num_layers, _ACT[activation]
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, out_features))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )
