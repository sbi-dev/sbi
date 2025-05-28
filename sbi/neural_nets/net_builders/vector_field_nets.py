# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import math
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from sbi.neural_nets.estimators.flowmatching_estimator import FlowMatchingEstimator
from sbi.neural_nets.estimators.score_estimator import (
    ConditionalScoreEstimator,
    SubVPScoreEstimator,
    VEScoreEstimator,
    VPScoreEstimator,
)
from sbi.utils.nn_utils import get_numel
from sbi.utils.sbiutils import (
    standardizing_net,
    z_standardization,
)
from sbi.utils.user_input_checks import check_data_device
from sbi.utils.vector_field_utils import VectorFieldNet


# ==================== Building Flow/Score Matching Estimators =========================
def build_flow_matching_estimator(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_y: Optional[str] = None,
    embedding_net: nn.Module = nn.Identity(),
    hidden_features: Union[Sequence[int], int] = 100,
    time_embedding_dim: int = 32,
    num_layers: int = 5,
    num_blocks: int = 5,
    num_heads: int = 4,
    mlp_ratio: int = 4,
    net: str | nn.Module = "ada_mlp",  # "mlp", "ada_mlp", or "transformer"
    **kwargs,
) -> FlowMatchingEstimator:
    """Builds a flow matching estimator with the given network.

    Args:
        batch_x: Batch of xs, used to infer dimensionality.
        batch_y: Batch of ys, used to infer dimensionality.
        embedding_net: Embedding network for batch_y.
        hidden_features: Number of hidden features in each layer (for MLP) or dimension
            of hidden features (for transformer).
        time_embedding_dim: Number of dimensions for time embedding.
        num_layers: Number of layers in the network (for MLP).
        num_blocks: Number of transformer blocks (for transformer).
        num_heads: Number of attention heads per block (for transformer).
        mlp_ratio: Ratio for MLP hidden dimension (for transformer).
        net: Type of architecture to use, either "mlp", "ada_mlp", or "transformer".
        **kwargs: Additional arguments for the network.

    Returns:
        A flow matching estimator.
    """
    # Check inputs and device
    check_data_device(batch_x, batch_y)

    # Z-score setup
    embedding_net_y = (
        standardizing_net(batch_y, z_score_y == "structured")
        if z_score_y
        else nn.Identity()
    )

    # Build network if not provided
    if net == "mlp":
        # Filter out AdaMLP-specific parameters
        mlp_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in [
                "condition_emb_dim",
                "mlp_ratio",
                "num_intermediate_mlp_layers",
                "adamlp_ratio",
            ]
        }
        vectorfield_net = build_standard_mlp_network(
            batch_x=batch_x,
            batch_y=batch_y,
            hidden_features=hidden_features,
            num_layers=num_layers,
            time_embedding_dim=time_embedding_dim,
            embedding_net=embedding_net_y,
            **mlp_kwargs,
        )
    elif net == "ada_mlp":
        vectorfield_net = build_adamlp_network(
            batch_x=batch_x,
            batch_y=batch_y,
            hidden_features=hidden_features,
            num_layers=num_layers,
            time_embedding_dim=time_embedding_dim,
            embedding_net=embedding_net_y,
            **kwargs,
        )
    elif net == "transformer":
        # For transformer, hidden_features must be an int
        hidden_dim = (
            hidden_features if isinstance(hidden_features, int) else hidden_features[0]
        )
        vectorfield_net = build_transformer_network(
            batch_x=batch_x,
            batch_y=batch_y,
            hidden_features=hidden_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            time_embedding_dim=time_embedding_dim,
            embedding_net=embedding_net,
            **kwargs,
        )
    else:
        if isinstance(net, nn.Module):
            vectorfield_net = net
        else:
            raise ValueError(f"Unknown architecture: {net}")

    # Create the flow matching estimator
    return FlowMatchingEstimator(
        net=vectorfield_net,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape,
        embedding_net=embedding_net,
    )


def build_score_matching_estimator(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = None,
    z_score_y: Optional[str] = None,
    embedding_net: nn.Module = nn.Identity(),
    sde_type: str = "ve",  # "vp", "subvp", or "ve"
    hidden_features: Union[Sequence[int], int] = 100,
    num_layers: int = 5,
    num_blocks: int = 5,
    num_heads: int = 4,
    mlp_ratio: int = 4,
    time_embedding_dim: int = 32,
    net: str | nn.Module = "ada_mlp",  # "mlp", "ada_mlp", or "transformer"
    **kwargs,
) -> ConditionalScoreEstimator:
    """Builds a score matching estimator with the given network.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        z_score_y: Whether to z-score ys passing into the network.
        embedding_net: Embedding network for batch_y.
        sde_type: SDE type for score estimator, one of "vp", "subvp", or "ve".
        hidden_features: Number of hidden features in each layer (for MLP) or dimension
            of hidden features (for transformer).
        num_layers: Number of layers in the network (for MLP).
        num_blocks: Number of transformer blocks (for transformer).
        num_heads: Number of attention heads per block (for transformer).
        mlp_ratio: Ratio for MLP hidden dimension (for transformer).
        time_embedding_dim: Number of dimensions for time embedding.
        net: Type of architecture to use, either "mlp", "ada_mlp", or "transformer".
        **kwargs: Additional arguments for the network.

    Returns:
        A score matching estimator.
    """
    # Check inputs and device
    check_data_device(batch_x, batch_y)

    # Build network if not provided
    if net == "mlp":
        # Filter out AdaMLP-specific parameters
        mlp_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in [
                "condition_emb_dim",
                "mlp_ratio",
                "num_intermediate_mlp_layers",
                "adamlp_ratio",
            ]
        }
        vectorfield_net = build_standard_mlp_network(
            batch_x=batch_x,
            batch_y=batch_y,
            hidden_features=hidden_features,
            num_layers=num_layers,
            time_embedding_dim=time_embedding_dim,
            embedding_net=embedding_net,
            **mlp_kwargs,
        )
    elif net == "ada_mlp":
        vectorfield_net = build_adamlp_network(
            batch_x=batch_x,
            batch_y=batch_y,
            hidden_features=hidden_features,
            num_layers=num_layers,
            time_embedding_dim=time_embedding_dim,
            embedding_net=embedding_net,
            **kwargs,
        )
    elif net == "transformer":
        # For transformer, hidden_features must be an int
        hidden_dim = (
            hidden_features if isinstance(hidden_features, int) else hidden_features[0]
        )
        vectorfield_net = build_transformer_network(
            batch_x=batch_x,
            batch_y=batch_y,
            hidden_features=hidden_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            time_embedding_dim=time_embedding_dim,
            embedding_net=embedding_net,
            **kwargs,
        )
    else:
        if isinstance(net, nn.Module):
            vectorfield_net = net
        else:
            raise ValueError(f"Unknown architecture: {net}")

    # Z-score setup
    mean_0, std_0 = z_standardization(batch_x, z_score_x == "structured")

    # Create input embeddings
    embedding_net_y = (
        standardizing_net(batch_y, z_score_y == "structured")
        if z_score_y
        else nn.Identity()
    )

    # Choose the appropriate score estimator based on SDE type
    if sde_type == "vp":
        estimator_cls = VPScoreEstimator
    elif sde_type == "subvp":
        estimator_cls = SubVPScoreEstimator
    elif sde_type == "ve":
        estimator_cls = VEScoreEstimator
    else:
        raise ValueError(f"Unknown SDE type: {sde_type}")

    # Create the score estimator
    return estimator_cls(
        net=vectorfield_net,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape,
        embedding_net=embedding_net_y,
        mean_0=mean_0,
        std_0=std_0,
    )


# ======= Time Embedding Shared Components =======


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding as used in Vaswani et al. (2017).

    Can be used for time embedding in both vector field nets.
    """

    def __init__(self, embed_dim: int = 16, max_freq: float = 0.01):
        """Initialize sinusoidal embedding.

        args:
            embed_dim: dimension of the embedding (must be even)
            max_freq: maximum frequency denominator
        """
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embedding dimension must be even")

        self.embed_dim = embed_dim
        self.max_freq = max_freq
        # compute frequency bands
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(max_freq) / embed_dim)
        )
        self.register_buffer("div_term", div_term)
        self.out_features = embed_dim

    def forward(self, t: Tensor) -> Tensor:
        """Embed time using transformer sinusoidal embeddings.

        args:
            t: time tensor of shape (batch_size, 1) or (batch_size,) or scalar ()

        returns:
            time embedding of shape (batch_size, embed_dim) or (embed_dim,)
            for scalar input
        """
        # handle scalar inputs (0-dim tensors)
        if t.ndim == 0:
            # create output for a single time point
            time_embedding = torch.zeros(self.embed_dim, device=t.device)
            time_embedding[0::2] = torch.sin(t * self.div_term)
            time_embedding[1::2] = torch.cos(t * self.div_term)
            return time_embedding.unsqueeze(0)

        # ensure time has the right shape for broadcasting
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        # create embeddings pe(pos, 2i) = sin(pos/1000^(2i/d))
        # pe(pos, 2i+1) = cos(pos/1000^(2i/d))
        time_embedding = torch.zeros(t.shape[:-1] + (self.embed_dim,), device=t.device)
        time_embedding[:, 0::2] = torch.sin(t * self.div_term)
        time_embedding[:, 1::2] = torch.cos(t * self.div_term)

        return time_embedding


class RandomFourierTimeEmbedding(nn.Module):
    """Gaussian random features for encoding time steps.

    This is to be used as a utility for score-matching."""

    def __init__(self, embed_dim=256, scale=30.0, learnable=True):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.embed_dim = embed_dim
        self.scale = scale
        if not learnable:
            self.register_buffer("W", torch.randn(embed_dim // 2) * scale)
        else:
            self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale)

    def forward(self, times: Tensor):
        times_proj = times[:, None] * self.W[None, :] * 2 * torch.pi
        embedding = torch.cat([torch.sin(times_proj), torch.cos(times_proj)], dim=-1)
        return torch.squeeze(embedding, dim=1)


# ======= Neural Network Blocks =======


class AdaMLPBlock(nn.Module):
    r"""Residual MLP block module with adaptive layer norm for conditioning.

    Arguments:
        hidden_dim: The dimensionality of the MLP block.
        cond_dim: The number of embedding features.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        mlp_ratio: int = 1,
        activation: Callable = nn.GELU,
        gate_activation: Callable = torch.nn.Tanh,
    ):
        super().__init__()

        self.ada_ln = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

        # Initialize the last layer to zero
        self.ada_ln[-1].weight.data *= 0.01
        self.ada_ln[-1].bias.data.zero_()

        # MLP block
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            activation(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

        self.gate_activation = gate_activation()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Arguments:
            h: The input tensor, with shape (B, D_hidden).
            yt: The embedding vector, with shape (B, D_emb).

        Returns:
            The output tensor, with shape (B, D_x).
        """

        shift_, scale_, gate_ = self.ada_ln(cond).chunk(3, dim=-1)
        gate_ = self.gate_activation(gate_)
        y = (scale_ + 1) * x + shift_
        y = self.block(y)
        y = x + gate_ * y

        return y


class GlobalEmbeddingMLP(nn.Module):
    """
    Global embedding MLP that outputs the conditioning embedding
    that is fed into the AdaMLPBlock.
    This MLP takes in the diffusion/flow timestep and the output from embedding net
    and outputs a single embedding vector.
    The timestep is a scalar for each batch, which is converted into
    either sinusoidal or random fourier embeddings.

    Args:
        cond_emb_dim: The dimensionality of the conditioning embedding.
        time_emb_type: Type of time embedding to use, "sinusoidal" or "random_fourier".
        time_emb_dim: The dimensionality of the time embedding.
        sinusoidal_max_freq: The maximum frequency for sinusoidal embeddings.
        fourier_scale: The scale for random fourier embeddings.
        hidden_dim: The dimensionality of the MLP block.
        num_intermediate_layers: Number of intermediate MLP blocks (Linear+GeLU+Linear).
        mlp_ratio: The ratio of the hidden dimension to the intermediate dimension.
        **kwargs: Key word arguments handed to the AdaMLPBlock.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        time_emb_type: str = "sinusoidal",
        time_emb_dim: int = 32,
        sinusoidal_max_freq: float = 0.01,
        fourier_scale: float = 30.0,
        hidden_dim: int = 100,
        num_intermediate_layers: int = 0,
        mlp_ratio: int = 1,
        activation: Callable = nn.GELU,
        use_x_emb: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_intermediate_layers = num_intermediate_layers

        if time_emb_type == "sinusoidal":
            self.time_emb = SinusoidalTimeEmbedding(
                embed_dim=time_emb_dim, max_freq=sinusoidal_max_freq
            )
        elif time_emb_type == "random_fourier":
            self.time_emb = RandomFourierTimeEmbedding(
                embed_dim=time_emb_dim, scale=fourier_scale
            )
        else:
            raise ValueError(f"Unknown time embedding type: {time_emb_type}")

        if use_x_emb:
            input_embed_dim = max(time_emb_dim, input_dim)
            self.input_layer = nn.Linear(input_embed_dim + time_emb_dim, hidden_dim)
        else:
            self.input_layer = nn.Linear(time_emb_dim, hidden_dim)

        self.mlp_blocks = nn.ModuleList()

        for _i in range(num_intermediate_layers):
            self.mlp_blocks.append(
                nn.Sequential(
                    activation(),
                    nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
                    activation(),
                    nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
                )
            )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, t: Tensor, x_emb: Optional[Tensor] = None) -> Tensor:
        t_emb = self.time_emb(t)

        try:
            # Pad x_embed by repeating if smaller than time_emb_dim
            pad_width = max(0, self.time_emb.embed_dim - x_emb.shape[-1])
            x_emb = torch.nn.functional.pad(x_emb, (0, pad_width), mode='constant')
            cond_emb = torch.cat([x_emb, t_emb], dim=-1) if x_emb is not None else t_emb
        except RuntimeError as e:
            shapes = f"x_emb shape: {x_emb.shape if x_emb is not None else 'None'},"
            shapes += f"t_emb shape: {t_emb.shape}"
            raise RuntimeError(
                f"Failed to concatenate embeddings with shapes {shapes}"
            ) from e

        cond_emb = self.input_layer(cond_emb)
        for mlp_block in self.mlp_blocks:
            cond_emb = mlp_block(cond_emb)
        return self.output_layer(cond_emb)


class VectorFieldMLP(VectorFieldNet):
    """MLP for vector field estimation"""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        time_emb_dim: int,
        hidden_features: int = 100,
        num_layers: int = 5,
        activation: Callable = nn.GELU,
        layer_norm: bool = True,
        skip_connections: bool = True,
        time_emb_type: str = "random_fourier",
        sinusoidal_max_freq: float = 1000.0,
        fourier_scale: float = 30.0,
    ):
        """Initialize vector field MLP.

        Args:
            input_dim: Dimension of the input (theta or state).
            condition_dim: Dimension of the conditioning variable.
            time_emb_dim: Dimension of the time embedding.
            hidden_features: Number of hidden features in each layer.
            num_layers: Number of layers in the network.
            activation: Activation function.
            layer_norm: Whether to use layer normalization.
            skip_connections: Whether to use skip connections.
            time_emb_type: Type of time embedding ("sinusoidal" or "random_fourier").
            sinusoidal_max_freq: Maximum frequency for sinusoidal embeddings.
            fourier_scale: Scale for random fourier embeddings.
        """
        super().__init__()

        self.skip_connections = skip_connections
        self.layer_norm = layer_norm

        # Input layers
        self.input_layer_theta = nn.Linear(input_dim, hidden_features)
        self.input_layer_x = nn.Linear(condition_dim, hidden_features)
        self.input_merge_layer = nn.Linear(
            hidden_features + hidden_features, hidden_features
        )

        # Time embedding
        if time_emb_type == "sinusoidal":
            self.time_emb = SinusoidalTimeEmbedding(
                embed_dim=time_emb_dim, max_freq=sinusoidal_max_freq
            )
        elif time_emb_type == "random_fourier":
            self.time_emb = RandomFourierTimeEmbedding(
                embed_dim=time_emb_dim, scale=fourier_scale
            )
        else:
            raise ValueError(f"Unknown time embedding type: {time_emb_type}")

        # Main network layers
        self.activation = activation()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_features, hidden_features))

        # Layer normalization if enabled
        if layer_norm:
            self.layers_norm = nn.ModuleList()
            for _ in range(num_layers):
                self.layers_norm.append(nn.LayerNorm(hidden_features))

        # Time embedding projection
        self.time_linear_layer = nn.Linear(time_emb_dim, hidden_features)

        # Output layer
        self.output_layer = nn.Linear(hidden_features, input_dim)
        nn.init.zeros_(self.output_layer.weight)

    def forward(self, theta: Tensor, x_emb_cond: Tensor, t: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Args:
            theta: Parameters (for FMPE) or state (for NPSE).
            x: Conditioning information.
            t: Time parameter embedding.

        Returns:
            Vector field evaluation at the provided points.
        """
        # Process inputs
        theta_emb = self.input_layer_theta(theta)
        x_emb = self.input_layer_x(x_emb_cond)
        h = self.input_merge_layer(
            self.activation(torch.cat([theta_emb, x_emb], dim=-1))
        )

        # Process time embedding
        t_emb = self.time_emb(t)
        t_emb = self.time_linear_layer(t_emb)

        # Main network forward pass
        h = self.activation(h)
        for i in range(len(self.layers)):
            h_old = h
            h = self.layers[i](h)
            h = self.activation(h)
            h += t_emb
            if self.skip_connections:
                h += h_old
            if self.layer_norm:
                h = self.layers_norm[i](h)

        return self.output_layer(h)


class VectorFieldAdaMLP(VectorFieldNet):
    """MLP for vector field estimation

    Architecture adapted from "Scalable Diffusion Models with Transformers"
    (Peebles & Xie, 2022) with adaptive layer norm for conditioning.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        condition_emb_dim: int,
        time_emb_dim: int,
        hidden_features: int = 100,
        num_layers: int = 5,
        global_mlp_ratio: int = 4,
        num_intermediate_mlp_layers: int = 0,
        adamlp_ratio: int = 4,
        activation: Callable = nn.GELU,
        time_emb_type: str = "sinusoidal",
        sinusoidal_max_freq: float = 1000.0,
        fourier_scale: float = 30.0,
    ):
        """Initialize vector field MLP.

        Args:
            input_dim: Dimension of the input (theta or state).
            condition_emb_dim: Dimension of the conditioning variable.
            time_emb_dim: Dimension of the time embedding.
            hidden_features: Number of hidden features in each layer.
            num_layers: Number of layers in the network.
            global_mlp_ratio: Ratio of the hidden dimension to the intermediate
                dimension in the global MLP.
            num_intermediate_mlp_layers: Number of intermediate MLP blocks
                (Linear+GeLU+Linear) in the global MLP.
            adamlp_ratio: Ratio of the hidden dimension to the intermediate
                dimension in the AdaMLPBlock.
            activation: Activation function.
            time_emb_type: Type of time embedding to use, "sinusoidal" or
                "random_fourier".
            sinusoidal_max_freq: The maximum frequency for sinusoidal embeddings.
            fourier_scale: The scale for random fourier embeddings.
        """
        super().__init__()

        self.layers = nn.ModuleList()

        # Global MLP for time and condition embedding
        self.global_mlp = GlobalEmbeddingMLP(
            input_dim=condition_dim,
            output_dim=condition_emb_dim,
            time_emb_dim=time_emb_dim,
            num_intermediate_layers=num_intermediate_mlp_layers,
            global_mlp_ratio=global_mlp_ratio,
            time_emb_type=time_emb_type,
            sinusoidal_max_freq=sinusoidal_max_freq,
            fourier_scale=fourier_scale,
            activation=activation,
        )
        self.input_dim = hidden_features
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_features))

        # Hidden layers
        for _ in range(num_layers):
            self.layers.append(
                AdaMLPBlock(
                    hidden_dim=hidden_features,
                    cond_dim=condition_emb_dim,
                    mlp_ratio=adamlp_ratio,
                    activation=activation,
                )
            )

        # Output layer
        self.layers.append(nn.Linear(hidden_features, input_dim, bias=False))
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, theta: Tensor, x_emb_cond: Tensor, t: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Args:
            theta: Parameters (for FMPE) or state (for NPSE).
            x: Conditioning information.
            t: Time parameter embedding.

        Returns:
            Vector field evaluation at the provided points.
        """

        h = theta

        # Get condition embedding
        cond_emb = self.global_mlp(t, x_emb=x_emb_cond)

        # Forward pass through MLP
        h = self.layers[0](h)  # input to hidden layer

        for layer in self.layers[1:-1]:  # hidden layers
            h = layer(h, cond_emb)

        h = self.layers[-1](h)  # hidden to output

        return h


class DiTBlock(nn.Module):
    """transformer block with adaptive layer norm for conditioning.

    Architecture adapted from "Scalable Diffusion Models with Transformers"
    (Peebles & Xie, 2022) with adaptive layer norm for conditioning.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int,
        mlp_ratio: int = 2,
        activation: Callable = nn.GELU,
    ):
        """Initialize dit transformer block.

        args:
            hidden_dim: dimension of hidden features
            cond_dim: dimension of conditioning features
            num_heads: number of attention heads
            mlp_ratio: ratio for mlp hidden dimension
            activation: activation function
        """
        super().__init__()

        # adaptive layer norm for attention
        self.ada_affine = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),  # 3 for attn, 3 for mlp
        )

        # initialize last layer to zero
        self.ada_affine[-1].weight.data.zero_()
        self.ada_affine[-1].bias.data.zero_()

        # attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            activation(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

        # layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Forward pass through the block.

        args:
            x: input tensor (b, d)
            cond: conditioning tensor (b, d_cond)

        returns:
            output tensor (b, d)
        """
        # get adaptive ln parameters
        ada_params = self.ada_affine(cond)
        attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate = (
            ada_params.chunk(6, dim=-1)
        )

        batch_size = x.shape[0]

        # Handle reshaping more carefully to preserve batch dimension
        attn_scale = attn_scale.view(batch_size, 1, -1)
        attn_shift = attn_shift.view(batch_size, 1, -1)
        attn_gate = attn_gate.view(batch_size, 1, -1)
        mlp_scale = mlp_scale.view(batch_size, 1, -1)
        mlp_shift = mlp_shift.view(batch_size, 1, -1)
        mlp_gate = mlp_gate.view(batch_size, 1, -1)

        # attention with adaptive ln
        x_norm = self.norm1(x)
        x_norm = x_norm * (attn_scale + 1) + attn_shift

        # self-attention
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_gate * attn_out

        # mlp with adaptive ln
        x_norm = self.norm2(x)
        x_norm = x_norm * (mlp_scale + 1) + mlp_shift

        # mlp
        mlp_out = self.mlp(x_norm)
        x = x + mlp_gate * mlp_out

        return x


class DiTBlockWithCrossAttention(nn.Module):
    """DiT block with cross-attention on conditioning and adaptive layernorm on time.

    This block implements a transformer block with:
    1. Self-attention on the input
    2. Cross-attention on the conditioning
    3. Adaptive layer normalization based on time embedding
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        time_emb_dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        activation: Callable = nn.GELU,
    ):
        super().__init__()

        # time embedding to adaptive layer norm parameters
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),  # 3 pairs of scale/shift
        )

        # initialize the last layer to zero
        self.time_mlp[-1].weight.data *= 0.1
        self.time_mlp[-1].bias.data.zero_()

        # self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # project conditioning to hidden dimension
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            activation(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

        # layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, cond_emb: Tensor, t_emb: Tensor) -> Tensor:
        """forward pass through the block.

        args:
            x: input tensor (b, d)
            cond_emb: conditioning tensor (b, d)
            t_emb: time embedding tensor (b, d_t)

        returns:
            output tensor (b, d)
        """
        # get adaptive ln parameters from time embedding
        time_params = self.time_mlp(t_emb)
        attn_scale, attn_shift, attn_gate, mlp_scale, mlp_shift, mlp_gate = (
            time_params.chunk(6, dim=-1)
        )

        # unsqueeze to broadcast to sequence dimension
        attn_scale = attn_scale.unsqueeze(1)
        attn_shift = attn_shift.unsqueeze(1)
        attn_gate = attn_gate.unsqueeze(1)
        mlp_scale = mlp_scale.unsqueeze(1)
        mlp_shift = mlp_shift.unsqueeze(1)
        mlp_gate = mlp_gate.unsqueeze(1)

        # self-attention with adaptive ln
        x_norm = self.norm1(x)
        x_norm = x_norm * (attn_scale + 1) + attn_shift

        # self-attention
        self_attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_gate * self_attn_out

        # cross-attention with adaptive ln
        x_norm = self.norm2(x)
        x_norm = x_norm * (mlp_scale + 1) + mlp_shift

        # cross-attention with conditioning (no adaptive ln here)

        # project conditioning to hidden dimension
        cond_emb = cond_emb.unsqueeze(-2)
        cond_emb = self.cond_proj(cond_emb)
        cond_emb = cond_emb.squeeze(-2)
        cross_attn_out, _ = self.cross_attn(query=x_norm, key=cond_emb, value=cond_emb)
        x = x + cross_attn_out

        # mlp with adaptive ln
        x_norm = self.norm3(x)
        x_norm = x_norm * (mlp_scale + 1) + mlp_shift

        # mlp
        mlp_out = self.mlp(x_norm)
        x = x + mlp_gate * mlp_out

        return x


class VectorFieldTransformer(VectorFieldNet):
    """Transformer for vector field estimation.

    This class implements a DiT-style transformer architecture
    for vector field estimation, using adaptive layer normalization
    for conditioning.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_features: int = 100,
        num_layers: int = 5,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        time_emb_dim: int = 32,
        time_emb_type: str = "sinusoidal",
        sinusoidal_max_freq: float = 1000.0,
        fourier_scale: float = 30.0,
        activation: Callable = nn.GELU,
        is_x_emb_seq: bool = False,
        **kwargs,
    ):
        """Initialize dit-style transformer vector field network.

        Args:
            input_dim: Dimension of input data (theta).
            condition_dim: Dimension of conditioning data (x).
            hidden_features: Dimension of hidden features.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio for mlp hidden dimension.
            time_emb_dim: Dimension of time embedding.
            time_emb_type: Type of time embedding ('sinusoidal' or 'fourier').
            sinusoidal_max_freq: Maximum frequency for sinusoidal embedding.
            fourier_scale: Scale for fourier embedding.
            activation: Activation function.
            is_x_emb_seq: Whether x_emb is a sequence (if true, cross-attention is
                used). NOTE: Effective usage is somewhat blocked by #1414
            **kwargs: Additional arguments.
        """
        super().__init__()

        self.layers = nn.ModuleList()

        # global embedding mlp for conditioning
        self.global_mlp = GlobalEmbeddingMLP(
            input_dim=condition_dim,
            output_dim=hidden_features,
            time_emb_dim=time_emb_dim,
            num_intermediate_layers=2,
            global_mlp_ratio=4,
            time_emb_type=time_emb_type,
            sinusoidal_max_freq=sinusoidal_max_freq,
            fourier_scale=fourier_scale,
            activation=activation,
            use_x_emb=(not is_x_emb_seq),  # x_emb is introduced here if not a sequence
        )

        self.is_x_emb_seq = is_x_emb_seq
        self.input_dim = input_dim
        self.hidden_features = hidden_features

        # input projection
        self.input_proj = nn.Conv1d(
            input_dim, input_dim * hidden_features, kernel_size=1, groups=input_dim
        )

        # transformer blocks
        self.blocks = nn.ModuleList([
            (
                DiTBlock(
                    hidden_dim=hidden_features,
                    cond_dim=hidden_features,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    activation=activation,
                )
                if not is_x_emb_seq
                else DiTBlockWithCrossAttention(
                    hidden_dim=hidden_features,
                    cond_dim=condition_dim,
                    time_emb_dim=hidden_features,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    activation=activation,
                )
            )
            for _ in range(num_layers)
        ])

        # output projection, reshape later
        self.output_proj = nn.Linear(hidden_features, 1)

    def forward(self, theta: Tensor, x_emb_cond: Tensor, t: Tensor) -> Tensor:
        """Forward pass through the transformer.

        Args:
            theta: parameters (for FMPE) or state (for NPSE)
            x: conditioning information
            t: time parameter embedding

        Returns:
            Vector field evaluation at the provided points
        """
        batch_size = theta.shape[0]

        # Get condition embedding
        cond_emb = self.global_mlp(
            t, x_emb=x_emb_cond if not self.is_x_emb_seq else None
        )

        # Project input to hidden dimension using a simpler approach
        h = theta.view(batch_size, self.input_dim, 1)  # [b, d, 1]
        h = h.expand(-1, -1, self.hidden_features)  # [b, d, h]

        # pass through transformer blocks
        for _, block in enumerate(self.blocks):
            if self.is_x_emb_seq:
                h = block(h, x_emb_cond, cond_emb)
            else:
                h = block(h, cond_emb)

        # project to output dimension
        h = self.output_proj(h)
        h = h.squeeze(-1)

        return h


# ======= Factory Functions =======


def build_adamlp_network(
    batch_x: Tensor,
    batch_y: Tensor,
    hidden_features: Union[Sequence[int], int] = 100,
    num_layers: int = 5,
    time_embedding_dim: int = 32,
    condition_emb_dim: int = 100,
    embedding_net: nn.Module = nn.Identity(),
    mlp_ratio: int = 4,
    num_intermediate_mlp_layers: int = 0,
    adamlp_ratio: int = 4,
    activation: Callable = nn.GELU,
    time_emb_type: str = "sinusoidal",
    sinusoidal_max_freq: float = 1000.0,
    fourier_scale: float = 30.0,
    **kwargs,
) -> VectorFieldAdaMLP:
    """Builds an adaptive vector field MLP network.

    Args:
        batch_x: Batch of xs, used to infer dimensionality.
        batch_y: Batch of ys, used to infer dimensionality.
        hidden_features: Number of hidden features in each layer.
        num_layers: Number of layers in the network.
        time_embedding_dim: Number of dimensions for time embedding.
        condition_emb_dim: Dimension of the conditioning embedding.
        embedding_net: Embedding network for batch_y.
        mlp_ratio: Ratio of hidden dim to intermediate dim in global MLP.
        num_intermediate_mlp_layers: Number of intermediate layers in global MLP.
        adamlp_ratio: Ratio of hidden dim to intermediate dim in AdaMLPBlock.
        activation: Activation function.
        time_emb_type: Type of time embedding ("sinusoidal" or "random_fourier").
        sinusoidal_max_freq: Max frequency for sinusoidal embeddings.
        fourier_scale: Scale for random fourier embeddings.

    Returns:
        An adaptive vector field MLP network.
    """
    del kwargs  # Unused

    # Check inputs and device
    check_data_device(batch_x, batch_y)

    # Get dimensions
    x_numel = get_numel(batch_x)
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    # Create time embedding dimension
    time_emb_dim = time_embedding_dim

    # Create the vector field network (MLP)
    if isinstance(hidden_features, int):
        hidden_dim = hidden_features
    else:
        hidden_dim = hidden_features[0] if len(hidden_features) > 0 else 256

    vectorfield_net = VectorFieldAdaMLP(
        input_dim=x_numel,
        condition_dim=y_numel,
        condition_emb_dim=condition_emb_dim,
        time_emb_dim=time_emb_dim,
        hidden_features=hidden_dim,
        num_layers=num_layers,
        global_mlp_ratio=mlp_ratio,
        num_intermediate_mlp_layers=num_intermediate_mlp_layers,
        adamlp_ratio=adamlp_ratio,
        activation=activation,
        time_emb_type=time_emb_type,
        sinusoidal_max_freq=sinusoidal_max_freq,
        fourier_scale=fourier_scale,
    )

    return vectorfield_net


def build_standard_mlp_network(
    batch_x: Tensor,
    batch_y: Tensor,
    hidden_features: Union[Sequence[int], int] = 100,
    num_layers: int = 5,
    time_embedding_dim: int = 32,
    embedding_net: nn.Module = nn.Identity(),
    activation: Callable = nn.GELU,
    layer_norm: bool = True,
    skip_connections: bool = True,
    time_emb_type: str = "random_fourier",
    sinusoidal_max_freq: float = 1000.0,
    fourier_scale: float = 30.0,
    **kwargs,
) -> VectorFieldMLP:
    """Builds a standard vector field MLP network.

    Args:
        batch_x: Batch of xs, used to infer dimensionality.
        batch_y: Batch of ys, used to infer dimensionality.
        hidden_features: Number of hidden features in each layer.
        num_layers: Number of layers in the network.
        time_embedding_dim: Number of dimensions for time embedding.
        embedding_net: Embedding network for batch_y.
        activation: Activation function.
        layer_norm: Whether to use layer normalization.
        skip_connections: Whether to use skip connections.
        time_emb_type: Type of time embedding ("sinusoidal" or "random_fourier").
        sinusoidal_max_freq: Maximum frequency for sinusoidal embeddings.
        fourier_scale: Scale for random fourier embeddings.
        **kwargs: Additional arguments.

    Returns:
        A vector field MLP network.
    """
    del kwargs  # Unused

    # Check inputs and device
    check_data_device(batch_x, batch_y)

    # Get dimensions
    x_numel = get_numel(batch_x)
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    # Create time embedding dimension
    time_emb_dim = time_embedding_dim

    # Create the vector field network (MLP)
    if isinstance(hidden_features, int):
        hidden_dim = hidden_features
    else:
        hidden_dim = hidden_features[0] if len(hidden_features) > 0 else 256

    vectorfield_net = VectorFieldMLP(
        input_dim=x_numel,
        condition_dim=y_numel,
        time_emb_dim=time_emb_dim,
        hidden_features=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        layer_norm=layer_norm,
        skip_connections=skip_connections,
        time_emb_type=time_emb_type,
        sinusoidal_max_freq=sinusoidal_max_freq,
        fourier_scale=fourier_scale,
    )

    return vectorfield_net


def build_transformer_network(
    batch_x: Tensor,
    batch_y: Tensor,
    hidden_features: int = 100,
    num_blocks: int = 5,
    num_heads: int = 4,
    mlp_ratio: int = 4,
    time_embedding_dim: int = 32,
    embedding_net: nn.Module = nn.Identity(),
    time_emb_type: str = "sinusoidal",
    sinusoidal_max_freq: float = 1000.0,
    fourier_scale: float = 30.0,
    activation: Callable = nn.GELU,
    is_x_emb_seq: bool = False,
    **kwargs,
) -> VectorFieldTransformer:
    """Builds a vector field transformer network.

    Args:
        batch_x: Batch of xs, used to infer dimensionality.
        batch_y: Batch of ys, used to infer dimensionality.
        hidden_features: Dimension of hidden features.
        num_blocks: Number of transformer blocks.
        num_heads: Number of attention heads per block.
        mlp_ratio: Ratio for MLP hidden dimension.
        time_embedding_dim: Number of dimensions for time embedding.
        embedding_net: Embedding network for batch_y.
        time_emb_type: Type of time embedding ("sinusoidal" or "fourier").
        sinusoidal_max_freq: Max frequency for sinusoidal embeddings.
        fourier_scale: Scale for random fourier embeddings.
        activation: Activation function.
        is_x_emb_seq: Whether x embedding is a sequence (uses cross-attention).

    Returns:
        A vector field transformer network.
    """
    del kwargs  # Unused

    # Check inputs and device
    check_data_device(batch_x, batch_y)

    # Get dimensions
    x_numel = get_numel(batch_x)
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    # Create the vector field network (Transformer)
    vectorfield_net = VectorFieldTransformer(
        input_dim=x_numel,
        condition_dim=y_numel,
        hidden_features=hidden_features,
        num_layers=num_blocks,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        time_emb_dim=time_embedding_dim,
        time_emb_type=time_emb_type,
        sinusoidal_max_freq=sinusoidal_max_freq,
        fourier_scale=fourier_scale,
        activation=activation,
        is_x_emb_seq=is_x_emb_seq,
    )

    return vectorfield_net
