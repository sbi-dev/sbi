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


def build_mean_flow(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal["none", "independent", "structured"] = "independent",
    z_score_y: Literal["none", "independent", "structured"] = "independent",
    hidden_features: int = 256,
    num_blocks: int = 5,
    num_steps: int = 2,
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
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed to `MeanFlowNet`.

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

    net = MeanFlowNet(
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
    )
