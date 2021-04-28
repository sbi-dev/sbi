from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn

import numpy as np
import torch
from torch.distributions import Distribution
from torch import Tensor, nn

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape, Array
from sbi.utils import del_entries
from sbi.utils.torchutils import (
    ScalarFloat,
    ensure_theta_batched,
    ensure_x_batched,
    atleast_2d_float32_tensor,
)

from sbi.vi.build_q import build_q, build_optimizer
from sbi.vi.divergence_optimizers import DivergenceOptimizer

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
        prior: Distribution,
        x_shape: torch.Size,
        device: str = "cpu",
        flow_paras: dict = {},
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
        kwargs = del_entries(locals(), entries=("self", "__class__", "flow_paras"))

        super().__init__(**kwargs)
        self._purpose = f"Variational Posterior approximation"
        self._flow_paras = flow_paras

        self._q = build_q(self._prior.event_shape, self._prior.support, **flow_paras)
        self._optimizer = build_optimizer(self, "elbo")
        self._loss = "elbo"

    @property
    def q(self):
        """Variational distributon that will be learned"""
        return self._q

    @q.setter
    def q(self, q: Distribution) -> None:
        """See `set_default_x`."""
        self._set_q(q)

    @property
    def optimizer(self):
        """ Divergence optimizer for variational inference"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: DivergenceOptimizer) -> None:
        """See `set_default_x`."""
        if not isinstance(optimizer, DivergenceOptimizer):
            raise ValueError(
                "This class relies on a DivergenceOptimizer which minimize some divergence."
            )
        self._optimier = optimizer

    def _set_q(self, q: Distribution) -> None:
        """ Sets the distributions and checks some important properties. """
        if not isinstance(q, Distribution):
            raise ValueError(
                "We only support PyTorch distributions, please wrap your distributions as one!"
            )
        if not hasattr(q, "parameters"):
            warn(
                "Your distributions has no parameters, you can give the parameters manually to train but may consider to 'parameterize' the distribution"
            )
        self._q = q

    def predictive(self, x: Tensor, method: str = "naive", num_samples: int = 10000):
        """ Estimates the predictive distributions log prob """
        return self.expectation(
            lambda theta: self.net.log_prob(x.repeat(num_samples, 1), context=theta),
            method=method,
            num_samples=num_samples,
        )

    def predictive_sample(self, shape: torch.Size()):
        thetas = self.sample(shape)
        xs = self.net.sample(context=thetas)
        return xs

    def expectation(self, f: Callable, method: str = "naive", num_samples: int = 10000):
        """Computes the expectation with respect to the posterior E_q[f(X)]

        Args:
        f: Function for which we will compute the expectation
        method: Method either naive (just using the variatioanl posterior), is
        (importance sampling) or psis (pareto smoothed importance sampling).
        """
        if method == "naive":
            return f(self.sample((num_samples,))).mean()
        elif method == "is":
            print("TODO implemented")
        elif method == "psis":
            print("TODO implement")
        else:
            raise NotImplementedError("We only have the methods naive, is and psis.")

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

        theta = ensure_theta_batched(torch.as_tensor(theta))

        # Select and check x to condition on..
        x = atleast_2d_float32_tensor(self._x_else_default_x(x))
        self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)

        with torch.set_grad_enabled(track_gradients):
            return self._q.log_prob(theta.to(self._device))

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

        # Select and check x to condition on..
        x = atleast_2d_float32_tensor(self._x_else_default_x(x))
        self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)

        if track_gradients:
            return self._q.rsample(sample_shape)
        else:
            return self._q.sample(sample_shape)

    def train(
        self,
        x: Optional[Array] = None,
        loss: str = "elbo",
        n_particles: Optional[int] = 128,
        learning_rate: float = 1e-2,
        min_num_iters: Optional[int] = 100,
        max_num_iters: Optional[int] = 1000,
        clip_value: Optional[float] = 5.0,
        retrain_from_scratch: bool = False,
        resume_training: bool = False,
        show_progress_bar: bool = True,
        **kwargs,
    ):

        # Init q and the optimizer if necessary
        if retrain_from_scratch:
            self._q = build_q(
                self._prior.event_shape, self._prior.support, **self._flow_paras
            )
            self._optimizer = build_optimizer(
                self, loss, lr=learning_rate, clip_value=clip_value, **kwargs
            )

        if resume_training or self._loss != loss or self._optimizer._loss_name != loss:
            self._optimizer = build_optimizer(
                self, loss, lr=learning_rate, clip_value=clip_value, **kwargs
            )
            self._loss = loss

        # Check context
        if x is None:
            x = atleast_2d_float32_tensor(self._x_else_default_x(x)).to(self._device)
        else:
            x = atleast_2d_float32_tensor(self._x_else_default_x(x)).to(self._device)

            self._ensure_single_x(x)
            self._ensure_x_consistent_with_default_x(x)
            self.set_default_x(x)

        # Optimize
        self._optimizer.update(locals())
        optimizer = self._optimizer
        if show_progress_bar:
            iters = tqdm(range(max_num_iters))
        else:
            iters = range(max_num_iters)

        for i in iters:
            l = optimizer.step(x).cpu().numpy()
            if show_progress_bar and i % 10 == 0:
                iters.set_description("Loss: " + str(np.round(l, 2)))
            if i > min_num_iters:
                if optimizer.converged() and show_progress_bar:
                    print(f"Converged with loss: {np.round(l, 2)}")
                    break
        k = round(float(optimizer.evaluate(x)), 3)
        if k > 1:
            warn(
                "The quality of the variational posterior seems to be bad, increase the training iterations or consider a different model!"
            )
        else:
            if show_progress_bar:
                print(
                    f"Quality Score: {k} (smaller values are good, should be below 1, mode collapse may still occured.)"
                )
        self._summary = optimizer.summary
        self._summary["Quality score"] = k
