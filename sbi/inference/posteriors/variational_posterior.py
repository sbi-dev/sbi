from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn

import numpy as np
import torch
from torch import Tensor, nn

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape
from sbi.utils import del_entries
from sbi.utils.torchutils import ScalarFloat, ensure_theta_batched, ensure_x_batched

import pyro
from pyro import distributions as dist
from pyro.distributions import transforms
from pyro.nn import AutoRegressiveNN


class VariationalPosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods, obtained with
    SNLE.<br/><br/>
    SNLE trains a neural network to approximate the likelihood $p(x|\theta)$. The
    `SNLE_Posterior` class performs variational inference to approximate posterior q(\theta|x) \approx p(\theta|x).
    Where $q$ is a inverse autoregressive normalizing flow. 
    """

    def __init__(
        self,
        method_family: str,
        neural_net: nn.Module,
        x_shape: torch.Size,
        prior,
        flow: str = "spline_autoregressive",
        device: str = "cpu",
    ):
        r"""
        Args:
            method_family: One of snpe, snl, snre_a or snre_b.
            neural_net: A classifier for SNRE, a density estimator for SNPE and SNL.
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            flow: Flow used for variational family one of: [iaf, planar, radial,
            affine_coupling, spline, spline_autoregressive, spline_coupling]
            device: Training device, e.g., cpu or cuda:0.
        """
        kwargs = del_entries(locals(), entries=("self", "__class__", "flow"))
        super().__init__(**kwargs)
        self._purpose = f"Variational Posterior approximation"
        self.q = build_flow(self._prior, type=flow)

    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False,
    ) -> Tensor:
        r"""
        Returns the log-probability of $q(\theta|x).$

        This corresponds to an normalized variational posterior log-probability.

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.

        Returns:
            `(len(θ),)`-shaped log-probability $\log(p(x|\theta) \cdot p(\theta))$.

        """

        if self.default_x != x:
            self.set_default_x(x)
            self.train()

        with torch.set_grad_enabled(track_gradients):
            return self.q.log_prob(theta.to(self._device))

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        track_gradients: bool = False,
    ) -> Tensor:
        r"""
        Return samples from variational posterior distribution $q(\theta|x)$.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            track_gradient: Wheater to reparamterize samples which enables to pass
                gradients through it.

        Returns:
            Samples from posterior.
        """
        if self.default_x is None:
            self.set_default_x(x.to(self._device))
            self.train()

        if track_gradients:
            return self.q.rsample(sample_shape)
        else:
            return self.q.sample(sample_shape)

    def train(
        self,
        x_obs=None,
        steps=1501,
        lr=1e-3,
        elbo_particels=64,
        exp_decay=0.9999,
        clip_value=1.0,
    ):
        modules = nn.ModuleList(
            [t for t in self.q.transforms[0].parts if isinstance(t, nn.Module)]
        )
        modules.train()
        if x_obs is None:
            x_obs = self.default_x
        self.set_default_x(x_obs)
        obs = x_obs[torch.randint(x_obs.size(0), (elbo_particels,))]
        optimizer = torch.optim.Adam(modules.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, exp_decay)
        for step in range(steps):
            optimizer.zero_grad()
            samples = self.q.rsample((elbo_particels,))
            log_q = self.q.log_prob(samples)
            log_ll = self.net.log_prob(obs, context=samples)
            log_prior = self._prior.log_prob(samples)
            loss = (
                log_q[torch.isfinite(log_q)].mean()
                - log_ll[torch.isfinite(log_ll)].mean()
                - log_prior[torch.isfinite(log_prior)].mean()
            )
            loss.backward()
            nn.utils.clip_grad_value_(modules.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
            self.q.clear_cache()
            if step % 100 == 0:
                print("Elbo:", loss.detach())
        modules.eval()


def link_to_support(support):
    if isinstance(support.base_constraint, torch.distributions.constraints.interval):
        lb = support.base_constraint.lower_bound
        ub = support.base_constraint.upper_bound
        interval_len = torch.abs(ub - lb)
        support_transform = transforms.ComposeTransform(
            [
                transforms.SigmoidTransform(),
                torch.distributions.transforms.AffineTransform(lb, interval_len),
            ]
        )
    else:
        raise NotImplementedError("Not implemented")
    return support_transform


def build_flow(
    prior,
    num_flows=10,
    type="spline_autoregressive",
    link_support=True,
    batch_norm=False,
    permute=False,
    **kwargs,
):
    dim = prior.shape()[0]
    support = prior.support
    base_dist = pyro.distributions.Normal(torch.zeros(dim), torch.ones(dim))
    flows = []
    for i in range(num_flows):
        flows.append(flow_block(dim, type, **kwargs))
        if batch_norm:
            flows.append(transforms.batchnorm(dim))
        if permute:
            flows.append(transforms.permute(dim))
        if link_support and i == num_flows - 1:
            flows.append(link_to_support(support))
    t = transforms.ComposeTransform(flows).with_cache()
    dist = pyro.distributions.TransformedDistribution(base_dist, [t])
    return dist


def flow_block(dim, type, **kwargs):
    if type.lower() == "iaf":
        flow = transforms.affine_autoregressive(dim, **kwargs)
    elif type.lower() == "planar":
        flow = transforms.planar(dim, **kwargs)
    elif type.lower() == "radial":
        flow = transforms.radial(dim, **kwargs)
    elif type.lower() == "affine_coupling":
        flow = transforms.affine_coupling(dim, **kwargs)
    elif type.lower() == "spline":
        flow = transforms.spline(dim, **kwargs)
    elif type.lower() == "spline_autoregressive":
        flow = transforms.spline_autoregressive(dim, **kwargs)
    elif type.lower() == "spline_coupling":
        flow = transforms.spline_coupling(dim, **kwargs)
    else:
        raise NotImplementedError()
    return flow
