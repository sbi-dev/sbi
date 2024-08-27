from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from sbi.neural_nets.estimators.score_estimator import (
    ConditionalScoreEstimator,
    GaussianFourierTimeEmbedding,
    SubVPScoreEstimator,
    VEScoreEstimator,
    VPScoreEstimator,
)
from sbi.utils.sbiutils import standardizing_net, z_score_parser, z_standardization
from sbi.utils.user_input_checks import check_data_device


class EmbedInputs(nn.Module):
    """Constructs input handler that optionally standardizes and/or
    embeds the input and conditioning variables, as well as the diffusion time
    embedding.
    """

    def __init__(self, embedding_net_x, embedding_net_y, embedding_net_t):
        """Initializes the input handler.

        Args:
            embedding_net_x: Embedding network for x.
            embedding_net_y: Embedding network for y.
            embedding_net_t: Embedding network for time.
        """
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y
        self.embedding_net_t = embedding_net_t

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> tuple:
        """Forward pass of the input layer.

        Args:
            inputs: theta (x), x (y), and diffusion time (t).

        Returns:
            Potentially standardized and/or embedded output.
        """

        return (
            self.embedding_net_x(x),
            self.embedding_net_y(y),
            self.embedding_net_t(t),
        )


def build_input_handler(
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
        Input handler that provides x, y, and time embedding, and optionally z-scores.
    """

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net_y = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net_y
        )
    embedding_net_t = GaussianFourierTimeEmbedding(t_embedding_dim)
    input_handler = EmbedInputs(
        embedding_net_x,
        embedding_net_y,
        embedding_net_t,
    )
    return input_handler


def build_score_estimator(
    batch_x: Tensor,
    batch_y: Tensor,
    sde_type: Optional[str] = "vp",
    score_net: Optional[Union[str, nn.Module]] = "mlp",
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    t_embedding_dim: int = 16,
    num_layers: int = 3,
    hidden_features: int = 50,
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

    # Default to variance-preserving SDE
    if sde_type is None:
        sde_type = "vp"

    input_handler = build_input_handler(
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
            input_handler,
            hidden_dim=hidden_features,
            num_layers=num_layers,
        )
    elif score_net == "ada_mlp":
        score_net = AdaMLP(
            x_numel,
            t_embedding_dim + y_numel,
            input_handler,
            hidden_dim=hidden_features,
            num_layers=num_layers,
        )
    elif score_net == "resnet":
        raise NotImplementedError
    elif isinstance(score_net, nn.Module):
        pass
    else:
        raise ValueError(f"Invalid score network: {score_net}")

    if sde_type == "vp":
        estimator = VPScoreEstimator
    elif sde_type == "ve":
        estimator = VEScoreEstimator
    elif sde_type == "subvp":
        estimator = SubVPScoreEstimator
    else:
        raise ValueError(f"SDE type: {sde_type} not supported.")

    input_shape = batch_x.shape[1:]
    condition_shape = batch_y.shape[1:]
    return estimator(
        score_net, input_shape, condition_shape, mean_0=mean_0, std_0=std_0, **kwargs
    )


class MLP(nn.Module):
    """Simple fully connected neural network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input_handler: nn.Module,
        hidden_dim: int = 100,
        num_layers: int = 5,
        activation: nn.Module = nn.GELU(),
        layer_norm: bool = True,
        skip_connection: bool = True,
    ):
        """Initializes the MLP.

        Args:
            input_dim: The dimensionality of the input tensor.
            output_dim: The dimensionality of the output tensor.
            input_handler: The input handler module.
            hidden_dim: The dimensionality of the hidden layers.
            num_layers: The number of hidden layers.
            activation: The activation function.
            layer_norm: Whether to use layer normalization.
            skip_connection: Whether to use skip connections.
        """
        super().__init__()

        self.input_handler = input_handler
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

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        x, y, t = self.input_handler(x, y, t)
        xyt = torch.cat([x, y, t], dim=-1)

        h = self.activation(self.layers[0](xyt))

        # Forward pass through hidden layers
        for i in range(1, self.num_layers - 1):
            h_new = self.layers[i](h)
            h = (h + h_new) if self.skip_connection else h_new

        # Output layer
        output = self.layers[-1](h)

        return output


class AdaMLPBlock(nn.Module):
    r"""Creates a residual MLP block module with adaptive layer norm for conditioning.

    Arguments:
        hidden_dim: The dimensionality of the MLP block.
        cond_dim: The number of embedding features.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        mlp_ratio: int = 1,
    ):
        super().__init__()

        self.ada_ln = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

        # Initialize the last layer to zero
        self.ada_ln[-1].weight.data.zero_()
        self.ada_ln[-1].bias.data.zero_()

        # MLP block
        # NOTE: This can be made more flexible to support layer types.
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, x: Tensor, yt: Tensor) -> Tensor:
        """
        Arguments:
            x: The input tensor, with shape (B, D_x).
            t: The embedding vector, with shape (B, D_t).

        Returns:
            The output tensor, with shape (B, D_x).
        """

        a, b, c = self.ada_ln(yt).chunk(3, dim=-1)

        y = (a + 1) * x + b
        y = self.block(y)
        y = x + c * y
        y = y / torch.sqrt(1 + c * c)

        return y


class AdaMLP(nn.Module):
    """
    MLP denoising network using adaptive layer normalization for conditioning.
    Relevant literature: https://arxiv.org/abs/2212.09748

    See "Scalable Diffusion Models with Transformers", by William Peebles, Saining Xie.

    Arguments:
        x_dim: The dimensionality of the input tensor.
        emb_dim: The number of embedding features.
        input_handler: The input handler module.
        hidden_dim: The dimensionality of the MLP block.
        num_layers: The number of MLP blocks.
        **kwargs: Key word arguments handed to the AdaMLPBlock.
    """

    def __init__(
        self,
        x_dim: int,
        emb_dim: int,
        input_handler: nn.Module,
        hidden_dim: int = 100,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.input_handler = input_handler
        self.num_layers = num_layers

        self.ada_blocks = nn.ModuleList()
        for _i in range(num_layers):
            self.ada_blocks.append(AdaMLPBlock(hidden_dim, emb_dim, **kwargs))

        self.input_layer = nn.Linear(x_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, x_dim)

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        x, y, t = self.input_handler(x, y, t)
        yt = torch.cat([y, t], dim=-1)

        h = self.input_layer(x)
        for i in range(self.num_layers):
            h = self.ada_blocks[i](h, yt)
        return self.output_layer(h)
