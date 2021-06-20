import torch
from torch import nn
import numpy as np


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ContinousTimeDenseScoreNet(nn.Module):
    def __init__(
        self,
        input_dim,
        marginal_prob,
        hidden_dim=100,
        nonlinearity=nn.ELU,
        context_dim=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.marginal_prob = marginal_prob
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.act = lambda x: x * torch.sigmoid(x)
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nonlinearity()
        )
        if context_dim is not None:
            self.context_embed = nn.Sequential(
                nn.Linear(context_dim, hidden_dim), nonlinearity()
            )
        self.layer1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nonlinearity()
        )
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nonlinearity()
        )
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nonlinearity()
        )
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nonlinearity()
        )
        self.dense4 = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t, context=None):
        # Encoding for time
        embed = self.act(self.time_embed(t.squeeze()))
        if self.context_dim is not None:
            context_embed = self.context_embed(context)
        # Encoding for input
        h1 = self.input_embed(x)
        if self.context_dim is not None:
            h1 = h1 + context_embed
        h1 += self.dense1(embed)
        h1 = self.act(h1)
        h2 = self.layer1(h1)
        h2 += self.dense2(embed)
        h2 = self.act(h2)
        h3 = self.layer2(h2)
        h3 += self.dense3(embed)
        h3 = self.act(h3)
        h4 = self.layer3(h3)
        h4 += self.dense4(embed)
        h4 = self.act(h4)
        h4 = self.layer4(h4)
        h = self.out_layer(h4)
        # Normalize output
        h = h / self.marginal_prob(t)[1]
        return h


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ContinousTimeConvolutionalScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(
        self,
        marginal_prob_std,
        channels=[32, 64, 128, 256],
        embed_dim=256,
        context_dim=None,
    ):
        """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h

