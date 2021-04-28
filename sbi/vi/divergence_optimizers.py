import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR

from typing import Optional, Iterable
import re

import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy

from sbi.vi.mixture_of_flows import Mixture
from sbi.vi.paretto_smoothed_is import gpdfit


def parameterize_distribution(q, paras):
    assert isinstance(paras, Iterable)

    def parameters(self):
        for para in paras:
            para.requires_grad_(True)
            yield para

    q.parameters = parameters
    return q


def filter_kwrags_for_func(f, kwargs):
    args = f.__code__.co_varnames
    new_kwargs = dict([(key, val) for key, val in kwargs.items() if key in args])
    return new_kwargs


class DivergenceOptimizer(ABC):
    r"""This is a wrapper around a PyTorch optimizer which is used to minimize some loss
    for variational inference.

    Args:
    posterior: Variational Posterior object.
    n_particles: Number of elbo_particels a.k.a the samples used to approximate the
    expectation.
    clip_value: Max value for gradient clipping.
    optimizer: PyTorch optimizer class, as default Adam is used.
    scheduler: PyTorch learning rate scheduler class, as default we use
    ExponentialLR with rate 1 (equivalent to use no scheduler).
    kwargs: All arguments associated with optimizer, scheduler and others
    """

    def __init__(
        self,
        posterior,
        n_particles: int = 128,
        clip_value: float = 5.0,
        optimizer: Optional[Optimizer] = Adam,
        scheduler: Optional[object] = ExponentialLR,
        **kwargs
    ):

        self.posterior = posterior
        self.n_particles = n_particles
        self.clip_value = clip_value
        self._kwargs = kwargs

        self.q = posterior._q
        self.likelihood = posterior.net
        self.prior = posterior._prior

        # Manage modules if present.
        if hasattr(self.q, "modules"):
            self.modules = nn.ModuleList(self.q.modules())
        elif "modules" in kwargs:
            self.modules = kwargs.pop("modules")
        else:
            self.modules = nn.ModuleList()
        self.modules.train()

        # Ensure that distribution has parameters
        if not hasattr(self.q, "parameters"):
            assert (
                "parameters" in kwargs
            ), "Your distribution has not parameters please give them to the optimizer!"
            parameters = kwargs.pop("parameters")
            parameterize_distribution(self.q, parameters)

        # Init optimizer and scheduler with correct arguments
        opt_kwargs = filter_kwrags_for_func(optimizer.__init__, kwargs)
        scheduler_kwargs = filter_kwrags_for_func(scheduler.__init__, kwargs)
        scheduler_kwargs["gamma"] = scheduler_kwargs.get("gamma", 1.0)
        self._optimizer = optimizer(self.q.parameters(), **opt_kwargs)
        self._scheduler = scheduler(self._optimizer, **scheduler_kwargs)
        self._scheduler._step_count = 2  # Prevents unnecessray warning...

        # Loss and summary
        self.loss_history = []
        self.summary = {"Moving average": np.array([]), "Moving std": np.array([])}

        # The loss that will be used
        self._loss = lambda x: torch.zeros(1), torch.zeros(1)
        # Hyperparameters to change adaptively
        self.HYPER_PARAMETERS = ["n_particles", "clip_value"]

    @abstractmethod
    def _generate_loss_function(self):
        """ This generates the loss function that will be used. """
        pass

    def evaluate(self, x_obs, N=int(1e5)):
        """ This will evaluate the posteriors quality """
        M = int(min(N / 5, 3 * np.sqrt(N)))
        x_obs = x_obs.repeat(N, 1)
        with torch.no_grad():
            samples = self.q.sample((N,))
            log_q = self.q.log_prob(samples)
            log_ll = self.likelihood.log_prob(x_obs, context=samples)
            log_prior = self.prior.log_prob(samples)
            logweights = log_ll + log_prior - log_q
            logweights_max = logweights.max()
            weights = torch.exp(logweights - logweights_max)
            vals, _ = weights.sort(descending=True)
            largest_weigths = vals[:M]
        k, _ = gpdfit(largest_weigths)
        return k

    def loss(self, x_obs):
        """Computes the loss function which is optimized.
        Args:
            x_obs: Observed data as input.

        Returns:
            surrogated_loss : The loss that will be differentiated, hence this must be
            differentiable by PyTorch
            loss : This loss will be displayed and used to determine convergence. This
            does not have to be differentiable.
        """
        return self._loss(x_obs)

    def step(self, x_obs):
        """ Performs one gradient step """
        self._optimizer.zero_grad()
        surrogate_loss, loss = self.loss(x_obs)
        surrogate_loss.backward()
        nn.utils.clip_grad_norm_(self.modules.parameters(), self.clip_value)
        self._optimizer.step()
        self.loss_history.append(float(loss))
        self._scheduler.step()
        return loss

    def converged(self, eps=1e-4, considered_values=100):
        mean, std = self.summarize()
        if len(mean) < considered_values:
            return False
        else:
            loss = np.array(self.loss_history)[-considered_values:]
            m1 = mean[-int(considered_values / 2) :].mean()
            m2 = mean[-considered_values : -int(considered_values / 2)].mean()
            s = std[-considered_values:].mean()
            return abs(m1 - m2) / s < eps

    def summarize(self, moving_window=10):
        """ Summarize training """
        loss = np.array(self.loss_history)
        N = min(moving_window, len(loss))
        moving_mean = np.convolve(loss, torch.ones(N) / N, mode="valid")
        moving_std = np.sqrt((loss[N - 1 :] - moving_mean) ** 2)
        self.summary["Moving average"] = moving_mean
        self.summary["Moving std"] = moving_std
        return moving_mean, moving_std

    def to(self, device: str):
        """ Moves parameters to device """
        for p in self.q.parameters():
            p.to(device)

    def update(self, kwargs: dict):
        """ Updates the hyperparameters """
        paras = self.__dict__
        for key, val in kwargs.items():
            if key in self.HYPER_PARAMETERS:
                paras[key] = val


class ElboOptimizer(DivergenceOptimizer):
    r"""This learns the variational posterior by minimizing the reverse KL
    divergence using the evidence lower bound (ELBO). This is done automatically if
    your variational distribution 'posterior.q' is a proper TransformedDistribution with a rsample method.

    If this is not the case, the optimizer expects a nn.Module class with a method
    'build_loss_elbo' which returns a function that gets x_obs and computes the
    elbo.

    Args:
    posterior: Variational Posterior object.
    reduce_variance: Reduce variance by only considering the pathwise derivative and
    dropping the score function as its expectation is zero...
    n_particles: Number of elbo_particels a.k.a the samples used to approximate the
    expectation.
    clip_value: Max value for gradient clipping.
    optimizer: PyTorch optimizer class, as default Adam is used.
    scheduler: PyTorch learning rate scheduler class, as default we use
    ExponentialLR with rate 1 (equivalent to use no scheduler).
    kwargs: All arguments associated with optimizer, scheduler and others
    """

    def __init__(self, *args, reduce_variance: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.reduce_variance = reduce_variance
        self._surrogate_q = deepcopy(self.q)

        self._generate_loss_function()
        self._loss_name = "elbo"
        self.HYPER_PARAMETERS += ["reduce_variance"]

    def _generate_loss_function(self):
        if self.q.has_rsample:
            self._loss = self.loss_rsample
        elif isinstance(self.q, Mixture):
            self._loss = self.loss_mixture
        else:
            raise NotImplementedError(
                "Currently only reparameterizable distributions or mixture of reparameterizable distributions are supported."
            )

    def loss_rsample(self, x_obs):
        """ Computes the elbo """
        elbo_particles = self.generate_elbo_particles(x_obs)
        loss = -elbo_particles.mean()
        return loss, loss.clone().detach()

    def loss_mixture(self, x_obs):
        mix = torch.softmax(self.q.mixture_distribution.logits, -1)
        mixture_particles = self.generate_mixture_particles(x_obs)
        loss = -mix @ mixture_particles.mean(0)
        return loss, loss.clone().detach()

    def generate_elbo_particles(self, x_obs, num_samples=None):
        """ Generates elbo particles """
        if num_samples is None:
            num_samples = self.n_particles
        x_obs = x_obs.repeat(num_samples, 1)
        samples = self.q.rsample((num_samples,))
        if self.reduce_variance:
            self.update_surrogate_q()
            log_q = self._surrogate_q.log_prob(samples)
        else:
            log_q = self.q.log_prob(samples)
        log_ll = self.likelihood.log_prob(x_obs, context=samples)
        log_prior = self.prior.log_prob(samples)
        elbo = log_ll + log_prior - log_q
        return elbo

    def generate_mixture_particles(self, x_obs, num_samples=None):
        if num_samples is None:
            num_samples = self.n_particles
        x_obs = x_obs.repeat(num_samples, self.q.num_components, 1)
        samples = self.q.rsample_components((num_samples,))
        if self.reduce_variance:
            self.update_surrogate_q()
            log_q = self._surrogate_q.log_prob(samples)
        else:
            log_q = self.q.log_prob(samples)
        log_ll = self.likelihood.log_prob(
            x_obs.reshape(-1, x_obs.shape[-1]),
            context=samples.reshape(-1, samples.shape[-1]),
        ).reshape(self.n_particles, self.q.num_components)
        log_prior = self.prior.log_prob(samples)
        elbo = log_ll + log_prior - log_q
        return elbo

    def update_surrogate_q(self):
        for param, param_surro in zip(
            self.q.parameters(), self._surrogate_q.parameters()
        ):
            param_surro.data = param.data
            param_surro.requires_grad = False


class IWElboOptimizer(ElboOptimizer):
    r"""This learns the variational posterior by minimizing the importance weighted elbo. This is done automatically if your variational distribution 'posterior.q' is a
    proper TransformedDistribution with a rsample method.

    If this is not the case, the optimizer expects a nn.Module class with a method
    'build_loss_renjey' which returns a function that gets x_obs and computes the loss.

    References:
    """

    def __init__(self, *args, K=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = K
        self.loss_name = "iwelbo"
        self.HYPER_PARAMETERS += ["K"]

    def loss_rsample(self, x_obs):
        """ Computes the elbo """
        elbo_particles = self.generate_elbo_particles(x_obs, self.n_particles * self.K)
        elbo_particles = elbo_particles.reshape(self.n_particles, self.K)
        weights = self.get_importance_weight(elbo_particles.clone().detach())
        surrogate_loss = -(weights * elbo_particles).sum(-1).mean(0)
        loss = -torch.mean(torch.exp(elbo_particles) + 1e-6, -1).log().mean()
        return surrogate_loss, loss.clone().detach()

    def loss_mixture(self, x_obs):
        mix = torch.softmax(self.q.mixture_distribution.logits, -1)
        mixture_particles = self.generate_mixture_particles(
            x_obs, self.n_particles * self.K * m
        )
        mixture_particles = mixture_particles.reshape(
            self.n_particles, self.K, self.q.num_components
        )
        loss = -mix @ mixture_particles.mean(0)
        return loss, loss.clone().detach()

    def get_importance_weight(self, elbo_particles):
        """ Computes the importance weights for the gradients """
        logweights = elbo_particles
        normalized_weights = torch.exp(
            logweights - torch.logsumexp(logweights, -1).unsqueeze(-1)
        )
        return normalized_weights


class RenjeyDivergenceOptimizer(ElboOptimizer):
    r"""This learns the variational posterior by minimizing alpha divergences. This is
    done automatically if your variational distribution 'posterior.q' is a proper
    TransformedDistribution with a rsample method.

    If this is not the case, the optimizer expects a nn.Module class with a method
    'build_loss_renjey' which returns a function that gets x_obs and computes the
    loss.

    References: https://arxiv.org/abs/1602.02311
    """

    def __init__(self, *args, alpha=0.5, **kwargs):
        """
        Args:
            posterior: Variational Posterior object.
            alpha: Fixes which alpha divergence is optimized.
            n_particles: Number of elbo_particels.
            optimizer: PyTorch optimizer class, as default Adam is used.
            clip_value: Max value for gradient clipping.
        """
        self.alpha = alpha
        super().__init__(*args, **kwargs)
        self.loss_name = "renjey_divergence"
        self.HYPER_PARAMETERS += ["alpha"]

    def _generate_loss_function(self):
        if isinstance(self.alpha, float):
            self._loss = self.loss_alpha
        elif isinstance(self.alpha, str):
            K = re.findall(r"\b\d+\b", self.alpha)
            if "max" in self.alpha:
                if len(K) > 0:
                    self._loss = lambda x: self.loss_max(
                        x, K=min(K[0], self.n_particles)
                    )
                else:
                    self._loss = self.loss_max
            if "min" in self.alpha:
                if len(K) > 0:
                    self._loss = lambda x: self.loss_min(
                        x, K=min(K[0], self.n_particles)
                    )
                else:
                    self._loss = self.loss_max
            if "rand" in self.alpha:
                if len(K) > 0:
                    self._loss = lambda x: self.loss_min(
                        x, K=min(K[0], self.n_particles)
                    )
                else:
                    self._loss = self.loss_max

    def loss_alpha(self, x_obs):
        """ Loss given a finite alpha """
        elbo_particles = self.generate_elbo_particles(x_obs)
        weights, mean_log_weights = self.get_importance_weight(
            elbo_particles.clone().detach()
        )
        surrogate_loss = -torch.mean(weights * elbo_particles)
        loss = -mean_log_weights / (1 - self.alpha)
        return surrogate_loss, loss

    def loss_max(self, x_obs, K=1):
        """ Loss for $$\alpha = -\infty$$, e.g. alpha='max'"""
        elbo_particles = self.generate_elbo_particles(x_obs)
        sorted_elbo_particles, _ = elbo_particles.sort()
        surrogate_loss = -sorted_elbo_particles[-K:].mean()
        return surrogate_loss, surrogate_loss.detach()

    def loss_min(self, x_obs, K=1):
        """ Loss for $$\alpha = \infty$$, e.g. alpha='min'"""
        elbo_particles = self.generate_elbo_particles(x_obs)
        sorted_elbo_particles, _ = elbo_particles.sort()
        loss = -sorted_elbo_particles[:K].mean()
        return loss, loss.detach()

    def loss_rand(self, x_obs, K=1):
        """Stochastic loss, used elbo_particles are drawn according to probability
        given by the normalized weights."""
        elbo_particles = self.generate_elbo_particles(x_obs)
        weights, mean_log_weights = self.get_importance_weight(
            elbo_particles.clone().detach()
        )
        normalized_weights = weights / weights.sum()
        indices = torch.multinomial(normalized_weights, K)
        loss = -elbo_particles[indices].mean()
        return loss.loss.clone().detach()

    def get_importance_weight(self, elbo_particles):
        """ Computes the importance weights for the gradients """
        logweights = (1 - self.alpha) * elbo_particles
        mean_log_weights = torch.logsumexp(logweights, 0) - np.log(self.n_particles)
        normed_logweights = logweights - mean_log_weights
        weights = normed_logweights.exp()
        return weights, mean_log_weights


class TailAdaptivefDivergenceOptimizer(ElboOptimizer):
    r"""This learns the variational posterior by minimizing tail adaptive f divergences.
    This is done automatically if your variational distribution 'posterior.q' is a
    proper TransformedDistribution with a rsample method.

    If this is not the case, the optimizer expects a nn.Module class with a method
    'build_loss_renjey' which returns a function that gets x_obs and computes the loss.

    References: https://arxiv.org/abs/1810.11943
    """

    def __init__(
        self,
        posterior,
        beta: float = -1.0,
        n_particles: int = 128,
        clip_value: float = 5.0,
        optimizer=torch.optim.Adam,
        **kwargs
    ):
        super().__init__(posterior, n_particles, clip_value, optimizer, **kwargs)
        self.beta = beta
        self.loss_name = "tail_adaptive_fdivergence"
        self.HYPER_PARAMETERS += ["beta"]

        # if hasattr(self.modules, "build_tail_adaptive_loss"):
        #     self.loss = self.modules.build_tail_adaptive_loss(self)

    def loss(self, x_obs):
        """ Computes adaptive f divergence loss """
        elbo_particles = self.generate_elbo_particles(x_obs)
        gammas = self.get_tail_adaptive_weights(elbo_particles)

        surrogate_loss = -torch.sum(torch.unsqueeze(gammas * elbo_particles, 1))

        return surrogate_loss, surrogate_loss.clone().detach()

    def get_tail_adaptive_weights(self, elbo_particles):
        """ Computes the tail adaptive weights """
        weights = torch.exp(elbo_particles - elbo_particles.max())
        prob = torch.sign(weights.unsqueeze(1) - weights.unsqueeze(0))
        prob = torch.greater(prob, 0.5).float()
        F = 1 - prob.sum(1) / self.n_particles
        gammas = F ** self.beta
        gammas /= gammas.sum()
        return gammas.clone().detach()


ElboOptimizer.__init__.__doc__ = Optimizer.__init__.__doc__
