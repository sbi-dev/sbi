# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import math
from typing import Literal

import torch
from torch import Tensor, nn

from sbi.neural_nets.estimators.mean_flow_estimator import MeanFlowEstimator
from sbi.utils.nn_utils import get_numel
from sbi.utils.sbiutils import standardizing_net, z_score_parser, z_standardization
from sbi.utils.user_input_checks import check_data_device


class MeanFlowNet(nn.Module):
    r"""Two-headed network for a conditional mean flow.

    A shared trunk of adaptive-layer-norm residual blocks feeds two heads with
    their own last blocks: the average velocity $u(z, s, t, x)$ and the
    instantaneous velocity $v(z, s, x)$, which is the trunk evaluated at the time
    pair $(s, s)$. The condition embedding and random Fourier features of $s$ and
    $t - s$ modulate every block. Both output layers start at zero, so an untrained
    network maps every input to itself.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_features: int = 256,
        num_blocks: int = 5,
        num_head_blocks: int = 2,
        time_emb_dim: int = 32,
        fourier_scale: float = 0.3,
    ):
        r"""Initialize the network.

        Args:
            input_dim: Dimension of the modeled variable.
            condition_dim: Dimension of the embedded condition.
            hidden_features: Width of every block.
            num_blocks: Number of residual blocks on the path to each output.
            num_head_blocks: Number of those blocks that each head has on its own.
            time_emb_dim: Number of Fourier features per time.
            fourier_scale: Standard deviation of the Fourier frequencies.
        """
        # Imported here: vector_field_nets imports estimator_configs, which
        # imports this module.
        from sbi.neural_nets.net_builders.vector_field_nets import AdaMLPBlock

        super().__init__()
        if not 0 < num_head_blocks < num_blocks:
            raise ValueError(
                "num_head_blocks must be positive and smaller than num_blocks, got "
                f"{num_head_blocks} and {num_blocks}."
            )
        if time_emb_dim % 2 != 0:
            raise ValueError(f"time_emb_dim must be even, got {time_emb_dim}.")
        self.register_buffer(
            "_frequencies", torch.randn(time_emb_dim // 2) * fourier_scale
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(condition_dim + 2 * time_emb_dim, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )
        self.input_layer = nn.Linear(input_dim + condition_dim, hidden_features)

        def blocks(n: int) -> nn.ModuleList:
            return nn.ModuleList(
                AdaMLPBlock(hidden_features, cond_dim=hidden_features) for _ in range(n)
            )

        self.trunk = blocks(num_blocks - num_head_blocks)
        self.heads = nn.ModuleDict({
            "mean": blocks(num_head_blocks),
            "velocity": blocks(num_head_blocks),
        })
        self.outputs = nn.ModuleDict({
            "mean": nn.Linear(hidden_features, input_dim),
            "velocity": nn.Linear(hidden_features, input_dim),
        })
        for output in self.outputs.values():
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(
        self,
        z: Tensor,
        condition_emb: Tensor,
        s: Tensor,
        t: Tensor,
        head: Literal["mean", "velocity"] = "mean",
    ) -> Tensor:
        r"""Return $u(z, s, t, x)$ for `head="mean"` or $v(z, s, x)$ otherwise.

        Args:
            z: Standardized modeled variable, shape `(batch_dim, input_dim)`.
            condition_emb: Embedded condition, shape `(batch_dim, condition_dim)`.
            s: Start times, shape `(batch_dim,)`.
            t: End times, shape `(batch_dim,)`. Ignored by the velocity head.
            head: Which output to return.

        Returns:
            Velocity of shape `(batch_dim, input_dim)`.
        """
        if head == "velocity":
            t = s
        time_emb = torch.cat([self._embed_time(s), self._embed_time(t - s)], dim=-1)
        modulation = self.global_mlp(torch.cat([condition_emb, time_emb], dim=-1))
        h = self.input_layer(torch.cat([z, condition_emb], dim=-1))
        for block in [*self.trunk, *self.heads[head]]:
            h = block(h, modulation)
        return self.outputs[head](h)

    def _embed_time(self, time: Tensor) -> Tensor:
        projection = 2 * math.pi * time[:, None] * self._frequencies
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


def _mish(x: Tensor) -> Tensor:
    """Mish written out: `F.mish`'s second derivative under `torch.func` is NaN for
    large inputs, and the NUTS gradient of the density needs it."""
    return x * torch.tanh(nn.functional.softplus(x))


def _he_normal_(linear: nn.Linear) -> nn.Linear:
    """Keras' `he_normal`: normal truncated at two standard deviations, zero bias."""
    std = math.sqrt(2.0 / linear.in_features) / 0.87962566103423978
    nn.init.trunc_normal_(linear.weight, std=std, a=-2 * std, b=2 * std)
    nn.init.zeros_(linear.bias)
    return linear


class _FiLMBlock(nn.Module):
    """BayesFlow's `ConditionalDenseBlock` with its `TimeMLP` defaults: dense layer,
    mish, FiLM by the context, residual connection, layer norm."""

    def __init__(self, width: int, context_dim: int):
        super().__init__()
        self.dense = _he_normal_(nn.Linear(width, width))
        self.to_gamma = _he_normal_(nn.Linear(context_dim, width))
        self.to_beta = _he_normal_(nn.Linear(context_dim, width))
        # Keras' LayerNormalization default epsilon.
        self.norm = nn.LayerNorm(width, eps=1e-3)

    def forward(self, h: Tensor, context: Tensor) -> Tensor:
        update = _mish(self.dense(h))
        update = (1 + self.to_gamma(context)) * update + self.to_beta(context)
        return self.norm(h + update)


class PaperMeanFlowNet(nn.Module):
    r"""The mean-flow network of Arruda et al. (2026), ported from their code.

    BayesFlow's `TimeMLP` as their `MeanFlow` configures it: the times enter as the
    pair $(s, s - t)$ through trainable Fourier features plus the raw times, $z$ and
    the condition are projected separately and merged, and every block is modulated
    by FiLM from the time features and the condition projection. The velocity head
    clones the last `num_head_blocks` blocks. Initializations follow Keras'
    defaults, so an untrained network is not the identity.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_features: int = 256,
        num_blocks: int = 5,
        num_head_blocks: int = 2,
        time_emb_dim: int = 32,
        fourier_scale: float = 0.3,
    ):
        r"""Initialize the network.

        Args:
            input_dim: Dimension of the modeled variable.
            condition_dim: Dimension of the embedded condition.
            hidden_features: Width of every block.
            num_blocks: Number of blocks on the path to each output.
            num_head_blocks: Number of those blocks that each head has on its own.
            time_emb_dim: Number of Fourier features per time.
            fourier_scale: Factor on the trainable Fourier frequencies.
        """
        super().__init__()
        if not 0 < num_head_blocks < num_blocks:
            raise ValueError(
                "num_head_blocks must be positive and smaller than num_blocks, got "
                f"{num_head_blocks} and {num_blocks}."
            )
        if time_emb_dim % 2 != 0:
            raise ValueError(f"time_emb_dim must be even, got {time_emb_dim}.")
        self.fourier_scale = fourier_scale
        # Keras' `random_normal` initializer has standard deviation 0.05.
        self.frequencies = nn.Parameter(0.05 * torch.randn(2, time_emb_dim // 2))
        self.x_proj = _he_normal_(nn.Linear(input_dim, hidden_features))
        self.c_proj = _he_normal_(nn.Linear(condition_dim, hidden_features))
        self.merge_proj = _he_normal_(nn.Linear(2 * hidden_features, hidden_features))
        context_dim = 2 + 2 * time_emb_dim + hidden_features

        def blocks(n: int) -> nn.ModuleList:
            return nn.ModuleList(
                _FiLMBlock(hidden_features, context_dim) for _ in range(n)
            )

        self.trunk = blocks(num_blocks - num_head_blocks)
        self.heads = nn.ModuleDict({
            "mean": blocks(num_head_blocks),
            "velocity": blocks(num_head_blocks),
        })
        self.outputs = nn.ModuleDict({
            "mean": nn.Linear(hidden_features, input_dim),
            "velocity": nn.Linear(hidden_features, input_dim),
        })
        for output in self.outputs.values():
            # Keras' Dense default: Glorot uniform, zero bias.
            nn.init.xavier_uniform_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(
        self,
        z: Tensor,
        condition_emb: Tensor,
        s: Tensor,
        t: Tensor,
        head: Literal["mean", "velocity"] = "mean",
    ) -> Tensor:
        r"""Return $u(z, s, t, x)$ for `head="mean"` or $v(z, s, x)$ otherwise.

        Args:
            z: Standardized modeled variable, shape `(batch_dim, input_dim)`.
            condition_emb: Embedded condition, shape `(batch_dim, condition_dim)`.
            s: Start times, shape `(batch_dim,)`.
            t: End times, shape `(batch_dim,)`. Ignored by the velocity head.
            head: Which output to return.

        Returns:
            Velocity of shape `(batch_dim, input_dim)`.
        """
        if head == "velocity":
            t = s
        times = torch.stack([s, s - t], dim=-1)
        projection = (
            2 * math.pi * self.fourier_scale * times[..., None] * self.frequencies
        )
        fourier = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        time_emb = torch.cat([times, fourier.flatten(-2)], dim=-1)

        condition_proj = self.c_proj(condition_emb)
        merged = torch.cat([self.x_proj(z), condition_proj], dim=-1)
        h = _mish(self.merge_proj(_mish(merged)))
        context = torch.cat([time_emb, condition_proj], dim=-1)
        for block in [*self.trunk, *self.heads[head]]:
            h = block(h, context)
        return self.outputs[head](h)


def build_mean_flow(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal["none", "independent", "structured"] = "independent",
    z_score_y: Literal["none", "independent", "structured"] = "independent",
    hidden_features: int = 256,
    num_blocks: int = 5,
    num_steps: int = 2,
    variant: Literal["sbi", "paper"] = "sbi",
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> MeanFlowEstimator:
    r"""Build a conditional mean flow $q(x \mid y)$.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and
            std over the entire batch, instead of per-dimension.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_features: Width of every residual block.
        num_blocks: Number of residual blocks on the path to each output, the last
            two of which belong to each head alone.
        num_steps: Default number of flow-map steps of `log_prob` and `sample`.
        variant: `"sbi"` for `MeanFlowNet`, `"paper"` for `PaperMeanFlowNet` with
            the cycle-loss condition swap of the paper's code.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed to the network.

    Returns:
        A `MeanFlowEstimator`.
    """
    check_data_device(batch_x, batch_y)
    if batch_x.dim() != 2:
        raise ValueError(
            f"The mean flow requires a 1D modeled variable, got shape "
            f"{tuple(batch_x.shape[1:])}."
        )
    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    mean_x, std_x = (
        z_standardization(batch_x, structured_x) if z_score_x_bool else (0.0, 1.0)
    )
    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    if variant not in ("sbi", "paper"):
        raise ValueError(f"variant must be 'sbi' or 'paper', got {variant!r}.")
    net_class = PaperMeanFlowNet if variant == "paper" else MeanFlowNet
    net = net_class(
        input_dim=batch_x.shape[1],
        condition_dim=get_numel(batch_y, embedding_net=embedding_net),
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        **kwargs,
    )
    return MeanFlowEstimator(
        net,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape,
        embedding_net=embedding_net,
        mean_input=mean_x,
        std_input=std_x,
        num_steps=num_steps,
        cycle_swap="roll" if variant == "paper" else "permute",
    )
