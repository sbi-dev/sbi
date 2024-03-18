from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from sbi.neural_nets.embedding_nets import GaussianFourierTimeEmbedding
from sbi.neural_nets.estimators.score_estimator import (
    ConditionalScoreEstimator,
    VEScoreEstimator,
    VPScoreEstimator,
    subVPScoreEstimator,
)
from sbi.utils.sbiutils import standardizing_net, z_score_parser, z_standardization
from sbi.utils.user_input_checks import check_data_device


class EmbedInputs(nn.Module):
    """Constructs input layer that concatenates (and optionally standardizes and/or
    embeds) the input and conditioning variables, as well as the diffusion time
    embedding.
    """

    def __init__(self, embedding_net_x, embedding_net_y, embedding_net_t):
        """Initializes the input layer.

        Args:
            embedding_net_x: Embedding network for x.
            embedding_net_y: Embedding network for y.
            embedding_net_t: Embedding network for time.
        """
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y
        self.embedding_net_t = embedding_net_t

    def forward(self, inputs: list) -> Tensor:
        """Forward pass of the input layer.

        Args:
            inputs: List containing raw theta, x, and diffusion time.

        Returns:
            Concatenated and potentially standardized and/or embedded output.
        """

        assert (
            isinstance(inputs, list) and len(inputs) == 3
        ), """Inputs to network must be a list containing raw theta, x, and 1d time."""

        embeddings = [
            self.embedding_net_x(inputs[0]),
            self.embedding_net_y(inputs[1]),
            self.embedding_net_t(inputs[2]),
        ]
        out = torch.cat(
            embeddings,
            dim=-1,
        )
        return out


def build_input_layer(
    batch_y: Tensor,
    t_embedding_dim: int,
    z_score_y: Optional[str] = "independent",
    embedding_net_x: nn.Module = nn.Identity(),
    embedding_net_y: nn.Module = nn.Identity(),
) -> nn.Module:
    """Builds input layer for vector field regression, including time embedding, and
    optionally z-scores.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        t_embedding_dim: Dimensionality of the time embedding.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        embedding_net_x: Optional embedding network for x.
        embedding_net_y: Optional embedding network for y.

    Returns:
        Input layer that concatenates x, y, and time embedding, optionally z-scores.
    """

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net_y = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net_y
        )
    embedding_net_t = GaussianFourierTimeEmbedding(t_embedding_dim)
    input_layer = EmbedInputs(
        embedding_net_x,
        embedding_net_y,
        embedding_net_t,
    )
    return input_layer


def build_score_estimator(
    batch_x: Tensor,
    batch_y: Tensor,
    sde_type: Optional[str] = "vp",
    score_net: Optional[Union[str, nn.Module]] = "mlp",
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    t_embedding_dim: int = 32,
    num_layers: int = 3,
    hidden_features: int = 100,
    embedding_net_x: nn.Module = nn.Identity(),
    embedding_net_y: nn.Module = nn.Identity(),
    **kwargs,
) -> ConditionalScoreEstimator:
    """Builds score estimator for score-based generative models.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        sde_type: SDE type used, which defines the mean and std functions. One of:
            - 'vp': Variance preserving.
            - 'subvp': Sub-variance preserving.
            - 've': Variance exploding.
            Defaults to 'vp'.
        score_net: Type of regression network. One of:
            - 'mlp': Fully connected feed-forward network.
            - 'resnet': Residual network (NOT IMPLEMENTED).
            -  nn.Module: Custom network
            Defaults to 'mlp'.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        t_embedding_dim: Embedding dimension of diffusion time. Defaults to 16.
        num_layers: Number of MLP hidden layers. Defaults to 3.
        hidden_features: Number of hidden units per layer. Defaults to 50.
        embedding_net_x: Embedding network for x. Defaults to nn.Identity().
        embedding_net_y: Embedding network for y. Defaults to nn.Identity().
        kwargs: Additional arguments that are passed by the build function for score
            network hyperparameters.


    Returns:
        ScoreEstimator object with a specific SDE implementation.
    """

    """Builds score estimator for score-based generative models."""
    check_data_device(batch_x, batch_y)

    mean_0, std_0 = z_standardization(batch_x, z_score_x == "structured")

    input_layer = build_input_layer(
        batch_y,
        t_embedding_dim,
        z_score_y,
        embedding_net_x,
        embedding_net_y,
    )

    # Infer the output dimensionalities of the embedding_net by making a forward pass.
    x_numel = embedding_net_x(batch_x).shape[1:].numel()
    y_numel = embedding_net_y(batch_y).shape[1:].numel()

    if score_net == "mlp":
        score_net = MLP(
            x_numel + y_numel + t_embedding_dim,
            x_numel,
            hidden_dim=hidden_features,
            num_layers=num_layers,
        )
    elif score_net == "resnet":
        raise NotImplementedError
    elif isinstance(score_net, nn.Module):
        pass
    else:
        raise ValueError(f"Invalid score network: {score_net}")

    if sde_type == 'vp':
        estimator = VPScoreEstimator
    elif sde_type == 've':
        estimator = VEScoreEstimator
    elif sde_type == 'subvp':
        estimator = subVPScoreEstimator
    else:
        raise ValueError(f"SDE type: {sde_type} not supported.")

    neural_net = nn.Sequential(input_layer, score_net)
    input_shape = batch_x.shape[1:]
    condition_shape = batch_y.shape[1:]
    return estimator(
        neural_net, input_shape, condition_shape, mean_0=mean_0, std_0=std_0, **kwargs
    )


class MLP(nn.Module):
    """Simple fully connected neural network."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=100,
        num_layers=5,
        activation=nn.GELU(),
        layer_norm=True,
        skip_connection=True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.activation = activation
        self.skip_connection = skip_connection

        # Initialize layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            if layer_norm:
                block = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    activation,
                )
            else:
                block = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation)
            self.layers.append(block)

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        h = self.activation(self.layers[0](x))

        # Forward pass through hidden layers
        for i in range(1, self.num_layers - 1):
            h_new = self.layers[i](h)
            h = (h + h_new) if self.skip_connection else h_new

        # Output layer
        output = self.layers[-1](h)

        return output
