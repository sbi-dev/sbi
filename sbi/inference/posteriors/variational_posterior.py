from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn

import numpy as np
import torch
from torch import Tensor, nn

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape
from sbi.utils import del_entries
from sbi.utils.torchutils import ScalarFloat, ensure_theta_batched, ensure_x_batched

from sbi.vi.pyro_flows import build_flow
from sbi.vi import (
    build_q,
    ElboOptimizer,
    RenjeyDivergenceOptimizer,
    TailAdaptivefDivergenceOptimizer,
)

import pyro
from pyro import distributions as dist
from pyro.distributions import transforms
from pyro.nn import AutoRegressiveNN

from tqdm import tqdm


class VariationalPosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods, obtained with
    SNLE.<br/><br/>
    SNLE trains a neural network to approximate the likelihood $p(x|\theta)$. The
    `SNLE_Posterior` class performs variational inference to approximate posterior $q(\theta|x) \approx p(\theta|x)$.
    Where $q$ is a distribution of a specific variational family e.g. a Normalizing flow. 
    """

    def __init__(
        self,
        method_family: str,
        neural_net: nn.Module,
        prior,
        x_shape: torch.Size,
        flow_paras: dict = {},
        device: str = "cpu",
    ):
        """
        Args:
            method_family: One of snpe, snl, snre_a or snre_b.
            neural_net: A classifier for SNRE, a density estimator for SNPE and SNL.
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            x_shape: Shape of a single simulator output.
            flow: Flow used for variational family one of: [iaf, planar, radial,
            affine_coupling, spline, spline_autoregressive, spline_coupling]
            device: Training device, e.g., cpu or cuda:0.
        """
        kwargs1 = del_entries(
            locals(), entries=("self", "__class__", "flow", "mof", "flow_paras")
        )

        super().__init__(**kwargs1)
        self._purpose = f"Variational Posterior approximation"
        self._flow_paras = flow_paras
        self._optimizer = None
        self._summary = dict()

        self.q = build_q(self._prior.event_shape, self._prior.support, **flow_paras)

    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False,
    ) -> Tensor:
        """
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
        """
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
        loss: str = "elbo",
        n_particles: Optional[int] = 128,
        learning_rate: float = 1e-2,
        min_num_iters: Optional[int] = 100,
        max_num_iters: Optional[int] = 1000,
        clip_max_norm: Optional[float] = 5.0,
        retrain_from_scratch: bool = False,
        **kwargs,
    ):

        if retrain_from_scratch:
            self.q = build_q(
                self._prior.event_shape, self._prior.support, **self._flow_paras
            )

        if x_obs is None:
            x_obs = self.default_x

        # Choose correct optimizer
        if loss.lower() == "elbo":
            opt_kwargs = self.__optimizer_args(ElboOptimizer, kwargs)
            optimizer = ElboOptimizer(
                self,
                n_particles=n_particles,
                lr=learning_rate,
                clip_value=clip_max_norm,
                **opt_kwargs,
            )
        elif loss.lower() in ["renjey_divergence", "alpha_divergence"]:
            opt_kwargs = self.__optimizer_args(RenjeyDivergenceOptimizer, kwargs)
            optimizer = RenjeyDivergenceOptimizer(
                self,
                n_particles=n_particles,
                lr=learning_rate,
                clip_value=clip_max_norm,
                **opt_kwargs,
            )
        elif loss.lower() in ["tail_adaptive_fdivergence"]:
            opt_kwargs = self.__optimizer_args(TailAdaptivefDivergenceOptimizer, kwargs)
            optimizer = TailAdaptivefDivergenceOptimizer(
                self,
                n_particles=n_particles,
                lr=learning_rate,
                clip_value=clip_max_norm,
                **opt_kwargs,
            )
        self._optimizer = optimizer

        iters = tqdm(range(max_num_iters))
        loss = []
        eps = 1e-3 
        # TODO rewrite convergence check
        shift = int(min_num_iters / 2)
        for i in iters:
            l = optimizer.step(x_obs).numpy()
            loss.append(l)
            iters.set_description("Loss: " + str(np.round(l, 2)))
            if i > min_num_iters and i % 10 == 0:
                previous_mean = np.mean(loss[i - 2 * shift : i - shift])
                current_mean = np.mean(loss[i - shift : i])
                if abs(previous_mean - current_mean) < eps:
                    print(f"\nConverged with loss {np.round(l, 2)}")
                    break
        self._summary = loss
        # TODO evaluation

    def __optimizer_args(self, optimizer, kwargs):
        opt_args = optimizer.__init__.__code__.co_varnames
        opt_kwargs = dict(
            [(key, val) for key, val in kwargs.items() if key in opt_args]
        )
        return opt_kwargs
