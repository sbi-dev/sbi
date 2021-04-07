import torch
from torch import nn


import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    r"""This is a wrapper around a PyTorch optimizer which is used to minimize some loss for variational inference.
    """

    def __init__(
        self,
        posterior,
        n_particles: int = 128,
        clip_value: float = 5.0,
        optimizer=torch.optim.Adam,
        **kwargs
    ):
        """
        Args:
            posterior: Variational Posterior object.
            n_particles: Number of elbo_particels
            optimizer: PyTorch optimizer class, as default Adam is used
            clip_value: Max value for gradient clipping
        """
        self.posterior = posterior
        self.n_particles = n_particles
        self.clip_value = clip_value
        self._kwargs = kwargs

        # Determine trainable modules 
        if isinstance(posterior.q, torch.distributions.TransformedDistribution):
            self.modules = nn.ModuleList(
                [t for t in posterior.q.transforms if isinstance(t, nn.Module)]
            )
        elif isinstance(posterior.q, nn.Module):
            self.modules = posterior.q
        else:
            raise NotImplementedError("Unknown type of variational posterior q")

        # Init optimizer with correct arguments
        opt_kwargs = self.__filter_kwrags_for_func(optimizer.__init__)
        self.optimizer = optimizer(self.modules.parameters(), **opt_kwargs)

    @abstractmethod
    def loss(self, x_obs):
        """ Computes the loss function which is optimized.
        Args:
            x_obs: Observed data as input.
        
        Returns:
            surrogated_loss : The loss that will be differentiated, hence this must be
            differentiable by PyTorch
            loss : This loss will be displayed and used to determine convergence. This
            does not have to be differentiable.
        """
        pass

    @abstractmethod
    def step(self, x_obs):
        """ Computes current gradient and performs one gradient step.
        
        Args:
            x_obs: Observed data.
        
        Returns:
            loss : The loss after the step.
        """
        pass

    def __filter_kwrags_for_func(self, f):
        args = f.__code__.co_varnames
        new_kwargs = dict(
            [(key, val) for key, val in self._kwargs.items() if key in args]
        )
        return new_kwargs


class ElboOptimizer(Optimizer):
    r"""This learns the variational posterior by minimizing the reverse KL
        divergence using the evidence lower bound (ELBO). This is done automatically if
        your variational distribution 'posterior.q' is a proper TransformedDistribution with a rsample method.
       
        If this is not the case, the optimizer expects a nn.Module class with a method 'build_loss_elbo' which returns a function that gets x_obs and computes the elbo.
    """

    def __init__(
        self,
        posterior,
        n_particles: int = 128,
        clip_value: float = 5.0,
        optimizer=torch.optim.Adam,
        **kwargs
    ):
        """
        Args:
            posterior: Variational Posterior object to fit
            n_particles: Number of elbo particels
            clip_value: Value for gradient clipping
            optimizer: Optimizer class, default is Adam.
        """
        super().__init__(
            posterior,
            n_particles=n_particles,
            clip_value=clip_value,
            optimizer=optimizer,
            **kwargs
        )

        # If a custom loss is provide use this one
        if hasattr(self.modules, "build_loss_elbo"):
            self.loss = self.modules.build_loss_elbo(self)

    def loss(self, x_obs):
        """ Computes the elbo """
        elbo_particles = self.generate_elbo_particles(x_obs, n_samples=self.n_particles)
        loss = -elbo_particles.mean()
        return loss, loss.clone().detach()

    def step(self, x_obs):
        """ Performs one gradient step """
        self.optimizer.zero_grad()
        surrogate_loss, loss = self.loss(x_obs)
        surrogate_loss.backward()
        nn.utils.clip_grad_norm_(self.modules.parameters(), self.clip_value)
        self.optimizer.step()
        return loss

    def generate_elbo_particles(self, x_obs, n_samples=1):
        """ Generates elbo particles """
        x_obs = x_obs.repeat(n_samples, 1)
        samples = self.posterior.q.rsample((n_samples,))
        log_q = self.posterior.q.log_prob(samples)
        log_ll = self.posterior.net.log_prob(x_obs, context=samples)
        log_prior = self.posterior._prior.log_prob(samples)
        elbo = log_ll + log_prior - log_q
        return elbo


class RenjeyDivergenceOptimizer(ElboOptimizer):
    r"""This learns the variational posterior by minimizing alpha divergences. This is
        done automatically if your variational distribution 'posterior.q' is a proper
        TransformedDistribution with a rsample method.
       
        If this is not the case, the optimizer expects a nn.Module class with a method
        'build_loss_renjey' which returns a function that gets x_obs and computes the
        loss.
        
        References: https://arxiv.org/abs/1602.02311
    """

    def __init__(
        self,
        posterior,
        alpha: float = 0.5,
        n_particles: int = 128,
        clip_value: float = 5.0,
        optimizer=torch.optim.Adam,
        **kwargs
    ):
        """
        Args:
            posterior: Variational Posterior object.
            alpha: Fixes which alpha divergence is optimized.
            n_particles: Number of elbo_particels.
            optimizer: PyTorch optimizer class, as default Adam is used.
            clip_value: Max value for gradient clipping.
        """
        super().__init__(posterior, n_particles, clip_value, optimizer, **kwargs)
        self.alpha = alpha
        # Set correct loss function.
        if hasattr(self.modules, "build_loss_renjey"):
            self.loss = self.modules.build_loss_renjey(self)
        elif isinstance(self.alpha, (int, long, float)):
            self.loss = self.loss_alpha
        elif self.alpha == "max":
            self.loss = self.loss_max
        elif self.alpha == "min":
            self.loss = self.loss_min
        else:
            raise NotImplementedError(
                "Either your modules specify the loss or alpha must be a float, 'max' or 'min'."
            )

    def loss_alpha(self, x_obs):
        """ Loss given a finite alpha """
        elbo_particles = self.generate_elbo_particles(x_obs, n_samples=self.n_particles)
        weights, mean_log_weights = self.get_importance_weight(
            elbo_particles.clone().detach()
        )
        surrogate_loss = -torch.mean(weights * elbo_particles)
        loss = -mean_log_weights / (1 - self.alpha)
        return surrogate_loss, loss

    def loss_max(self, x_obs):
        """ Loss for $$\alpha = -\infty$$, e.g. alpha='max'"""
        elbo_particles = self.generate_elbo_particles(x_obs, n_samples=self.n_particles)
        loss = -elbo_particles.max()
        return loss, loss.detach()

    def loss_min(self, x_obs):
        """ Loss for $$\alpha = \infty$$, e.g. alpha='min'"""
        elbo_particles = self.generate_elbo_particles(x_obs, n_samples=self.n_particles)
        loss = -elbo_particles.min()
        return loss, loss.detach()

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

        if hasattr(self.modules, "build_tail_adaptive_loss"):
            self.loss = self.modules.build_tail_adaptive_loss(self)

    def loss(self, x_obs):
        """ Computes adaptive f divergence loss """
        elbo_particles = self.generate_elbo_particles(x_obs, n_samples=self.n_particles)
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
