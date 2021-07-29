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
from sbi.vi.sampling import gpdfit, clamp_weights, paretto_smoothed_weights


def parameterize_distribution(q, paras):
    assert isinstance(paras, Iterable)

    def parameters():
        for para in paras:
            para.requires_grad_(True)
            yield para

    q.parameters = parameters
    return q


def filter_kwrags_for_func(f, kwargs):
    args = f.__code__.co_varnames
    new_kwargs = dict([(key, val) for key, val in kwargs.items() if key in args])
    return new_kwargs


def make_sure_nothing_in_cache(q):
    """ This may be used before a 'deepcopy' call, as non leaf tensors (which are in the
    cache) do not support the deepcopy protocol..."""
    q.clear_cache()
    # The original methods can miss some parts..
    for t in q.transforms:
        t._cached_x_y = None, None
        # Compose transforms are not cleared correctly using q.clear_cache...
        if isinstance(t, torch.distributions.transforms.IndependentTransform):
            t = t.base_transform
        if isinstance(t, torch.distributions.transforms.ComposeTransform):
            for t_i in t.parts:
                t_i._cached_x_y = None, None

        t_dict = t.__dict__
        for key in t_dict:
            if "cache" in key or "det" in key:
                obj = t_dict[key]
                if torch.is_tensor(obj):
                    t_dict[key] = torch.zeros_like(obj)


class LikelihoodPosteriorWrapper:
    def __init__(self, posterior):
        self.posterior = posterior

    def log_prob(self, inputs, context=None):
        """This returns the ratio. Adding the prior gives the posterior and thus it
            behaves as the likelihood function up to normalizing constant
        
        
        
        Args:
            inputs: Samples for which log probability is evaluated
            context: Context on which we condition the samples
        
        Returns:
            tensor : log likelihood
        
        """
        self.posterior.net.eval()
        ll = self.posterior._log_likelihoods_over_trials(
            x=inputs, theta=context, net=self.posterior.net, track_gradients=True
        )
        return ll.squeeze()


class RatioBasedPosteriorWrapper:
    def __init__(self, posterior):
        self.posterior = posterior

    def log_prob(self, inputs, context=None):
        """This returns the ratio. Adding the prior gives the posterior and thus it
            behaves as the likelihood function up to normalizing constant, which drops
            by differentiation leading to equivalent gradients.
        
        
        
        Args:
            inputs: Samples for which log probability is evaluated
            context: Context on which we condition the samples
        
        Returns:
            [type]: [description]
        
        """
        self.posterior.net.eval()
        ratio = self.posterior._log_ratios_over_trials(
            x=inputs, theta=context, net=self.posterior.net, track_gradients=True
        )
        return ratio.squeeze()


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
    eps: This value determines the sensitivity of the convergence checks.
    kwargs: All arguments associated with optimizer, scheduler and others
    """

    def __init__(
        self,
        posterior,
        n_particles: int = 128,
        clip_value: float = 5.0,
        optimizer: Optional[Optimizer] = Adam,
        scheduler: Optional[object] = ExponentialLR,
        eps: float = 1e-5,
        **kwargs
    ):

        self.posterior = posterior
        self.n_particles = n_particles
        self.clip_value = clip_value
        self.device = posterior._device
        self.learning_rate = kwargs.get("lr", 1e-3)
        self._kwargs = kwargs

        # For convenience these get good names
        self.q = posterior._q
        self.set_likelihood_fn(posterior)
        self.prior = posterior._prior

        # This prevents error through optimization
        self.q.set_default_validate_args(False)
        self.prior.set_default_validate_args(False)

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
        for para in self.q.parameters():
            para.to(self.device)

        # Keep a state to resolve invalid values
        self.state_dict = [para.data.clone() for para in self.q.parameters()]

        # Init optimizer and scheduler with correct arguments
        opt_kwargs = filter_kwrags_for_func(optimizer.__init__, kwargs)
        scheduler_kwargs = filter_kwrags_for_func(scheduler.__init__, kwargs)
        scheduler_kwargs["gamma"] = scheduler_kwargs.get("gamma", 1.0)
        self._optimizer = optimizer(self.q.parameters(), **opt_kwargs)
        self._scheduler = scheduler(self._optimizer, **scheduler_kwargs)
        self._scheduler._step_count = 2  # Prevents unecessray warning...

        # Loss and summary
        self.eps = eps
        self.num_step = 0
        self.warm_up_was_done = False
        self.losses = np.ones(2000)
        self.moving_average = np.ones(2000)
        self.moving_std = np.ones(2000)
        self.moving_slope = np.ones(2000)

        # The loss that will be used
        self._loss = lambda x: torch.zeros(1), torch.zeros(1)
        # Hyperparameters to change adaptively
        self.HYPER_PARAMETERS = ["n_particles", "clip_value", "eps"]

    @abstractmethod
    def _generate_loss_function(self):
        """ This generates the loss function that will be used. """
        pass

    def set_likelihood_fn(self, posterior):
        if "snle" in posterior._method_family:
            self.likelihood = LikelihoodPosteriorWrapper(posterior)
        elif "snre" in posterior._method_family:
            self.likelihood = RatioBasedPosteriorWrapper(posterior)
        else:
            raise NotImplementedError("VI is only implemented for SNLE and SNRE")

    def to(self, device):
        """ This will move all parameters to the correct device, both for likelihood and
       posterior """
        for para in self.q.parameters():
            para.to(device)

    def warm_up(self, num_steps, method="prior"):
        """ This initializes q, either to follow the prior or the base distribution
      of the flow. This can increase training stability! """
        if method == "prior":
            p = self.prior
        elif method == "identity":
            p = torch.distributions.TransformedDistribution(
                self.q.base_dist, self.q.transforms[-1]
            )
        else:
            NotImplementedError("We only implement methods 'prior' or 'identity'")
        for _ in range(num_steps):
            self._optimizer.zero_grad()
            if self.q.has_rsample:
                samples = self.q.rsample((32,))
                logq = self.q.log_prob(samples)
                logp = p.log_prob(samples)
                loss = -torch.mean(logp - logq)
            else:
                samples = p.sample((256,))
                loss = -torch.mean(self.q.log_prob(samples))
            loss.backward()
            self._optimizer.step()
        # Denote that warmup was already done
        self.warm_up_was_done = True

    def update_state(self):
        """ This updates the current state. """
        for state_para, para in zip(self.state_dict, self.q.parameters()):
            if torch.isfinite(para).all():
                state_para.data = para.data.clone()
            else:
                nn.init.uniform_(para, a=-0.5, b=0.5)

    def resolve_state(self, warm_up_rounds=200):
        """ In case the parameters become nan, this method will try to fix the current state """
        for state_para, para in zip(self.state_dict, self.q.parameters()):
            para.data = state_para.data.clone().to(para.device)
        self._optimizer.__init__(self.q.parameters(), self.learning_rate)
        self.warm_up(warm_up_rounds)

    def evaluate(self, x_obs, N=int(5e4)):
        """ This will evaluate the posteriors quality """
        M = int(min(N / 5, 3 * np.sqrt(N)))
        with torch.no_grad():
            samples = self.q.sample((N,))
            log_q = self.q.log_prob(samples)
            log_ll = self.likelihood.log_prob(x_obs, context=samples)
            log_prior = self.prior.log_prob(samples)
            logweights = log_ll + log_prior - log_q
            logweights = logweights[torch.isfinite(logweights)]
            logweights_max = logweights.max()
            weights = torch.exp(logweights - logweights_max)
            vals, _ = weights.sort()
            largest_weigths = vals[-M:]
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
        surrogate_loss, loss = self.loss(x_obs.to(self.device))
        surrogate_loss.backward()
        if not torch.isfinite(surrogate_loss):
            self.resolve_state()
            return loss
        nn.utils.clip_grad_norm_(self.q.parameters(), self.clip_value)
        self._optimizer.step()
        self._scheduler.step()
        self.update_loss_stats(loss)
        if (self.num_step % 50) == 0:
            self.update_state()

    def converged(self, considered_values=50):
        """ Checks if the loss converged """
        if self.num_step < considered_values:
            return False
        else:
            m = self.moving_slope[
                self.num_step - considered_values : self.num_step
            ].mean()
            return abs(m) < self.eps

    def reset_loss_stats(self):
        self.losses = np.ones(2000)
        self.moving_average = np.ones(2000)
        self.moving_std = np.ones(2000)
        self.moving_slope = np.ones(2000)
        self.num_step = 0

    def update_loss_stats(self, loss, window=20):
        """Updates current loss statistics of the optimizer
    
        Args:
            loss: New loss value
            window: Window size for running statistics
        """
        if self.num_step >= len(self.losses):
            self.losses = np.append(self.losses, np.zeros(2000))
            self.moving_average = np.append(self.moving_average, np.zeros(2000))
            self.moving_std = np.append(self.moving_std, np.zeros(2000))
            self.moving_slope = np.append(self.moving_slope, np.zeros(2000))

        if self.num_step == 0:
            self.losses[self.num_step] = loss
            self.moving_average[self.num_step] = loss
        elif not np.isfinite(loss):
            loss = self.losses[self.num_step - 1]
            self.moving_average[self.num_step] = self.moving_average[self.num_step - 1]
            self.moving_std[self.num_step] = self.moving_std[self.num_step - 1]
            self.moving_slope[self.num_step] = self.moving_slope[self.num_step - 1]
        else:
            self.losses[self.num_step] = loss
            self.moving_average[self.num_step] = (
                self.moving_average[self.num_step - 1]
                + (loss - self.losses[max(self.num_step - window, 0)]) / window
            )
            self.moving_std[self.num_step] = abs(
                self.moving_std[self.num_step - 1]
                + (loss - self.losses[max(self.num_step - window, 0)]) / window
                - self.moving_std[self.num_step - 1] / (window - 1)
            )
            self.moving_slope[self.num_step] = (
                loss - self.moving_average[max(self.num_step - window, 0)]
            ) / window

        self.num_step += 1

    def get_loss_stats(self):
        """ Returns current loss statistics """
        return (
            self.moving_average[self.num_step - 1],
            self.moving_std[self.num_step - 1],
        )

    def update(self, kwargs: dict):
        """ Updates the hyperparameters and scheduler/optimizer kwargs"""
        paras = self.__dict__
        for key, val in kwargs.items():
            if key in self.HYPER_PARAMETERS:
                paras[key] = val

            if key == "self":
                posterior = kwargs[key]
                self.posterior = posterior
                self.q = posterior._q
                self.set_likelihood_fn(posterior)
                self.prior = posterior._prior
        opt_kwargs = filter_kwrags_for_func(type(self._optimizer).__init__, kwargs)
        scheduler_kwargs = filter_kwrags_for_func(
            type(self._scheduler).__init__, kwargs
        )
        scheduler_kwargs["gamma"] = scheduler_kwargs.get("gamma", 1.0)
        self._optimizer = type(self._optimizer)(self.q.parameters(), **opt_kwargs)
        self._scheduler = type(self._scheduler)(self._optimizer, **scheduler_kwargs)
        self._scheduler._step_count = 2


class ElboOptimizer(DivergenceOptimizer):
    r"""This learns the variational posterior by minimizing the reverse KL
    divergence using the evidence lower bound (ELBO)..

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
        make_sure_nothing_in_cache(self.q)
        self._surrogate_q = deepcopy(self.q)

        self._generate_loss_function()
        self._loss_name = "elbo"
        self.HYPER_PARAMETERS += ["reduce_variance"]
        self.eps = 1e-5

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
        """ Equivalent loss for mixtures with reparameterizable components"""
        mix = torch.softmax(self.q.mixture_distribution.logits, -1)
        mixture_particles = self.generate_mixture_particles(x_obs)
        loss = -mix @ mixture_particles.mean(0)
        return loss, loss.clone().detach()

    def generate_elbo_particles(self, x_obs, num_samples=None):
        """ Generates elbo particles """
        if num_samples is None:
            num_samples = self.n_particles
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
        """ Generates elbo particles for mixture distributions """
        if num_samples is None:
            num_samples = self.n_particles
        x_obs = x_obs.repeat(self.q.num_components, 1)
        samples = self.q.rsample_components((num_samples,))
        if self.reduce_variance:
            self.update_surrogate_q()
            log_q = self._surrogate_q.log_prob(samples)
        else:
            log_q = self.q.log_prob(samples)
        log_ll = self.likelihood.log_prob(
            x_obs.reshape(-1, x_obs.shape[-1]),
            context=samples.reshape(-1, samples.shape[-1]),
        ).reshape(num_samples, self.q.num_components)
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
    r"""This learns the variational posterior by minimizing the importance weighted
    elbo, which is an tighter bound to the evidence but also promotes a support covering behaviour.

    References:
    https://arxiv.org/abs/1509.00519

    NOTE: You may want to turn on 'reduce_variance' here as this loss leads to gradient
    estimates with vanishing SNR. This is relevant for large K, especially K > n_particles.
    """

    def __init__(self, *args, K=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = K
        self.loss_name = "iwelbo"
        self.HYPER_PARAMETERS += ["K"]
        self.eps = 5e-7

    def loss_rsample(self, x_obs):
        """ Computes the IWElbo """
        elbo_particles = self.generate_elbo_particles(x_obs, self.n_particles * self.K)
        elbo_particles = elbo_particles.reshape(self.n_particles, self.K)
        weights = self.get_importance_weight(elbo_particles.clone().detach())
        surrogate_loss = -(weights * elbo_particles).sum(-1).mean(0)
        loss = -torch.mean(torch.exp(elbo_particles) + 1e-6, -1).log().mean()
        return surrogate_loss, loss.clone().detach()

    def loss_mixture(self, x_obs):
        """ Computes the IWElbo """
        mix = torch.softmax(self.q.mixture_distribution.logits, -1).unsqueeze(0)
        mixture_particles = self.generate_mixture_particles(
            x_obs, num_samples=self.n_particles * self.K
        )
        mixture_particles = mixture_particles.reshape(
            self.n_particles, self.K, self.q.num_components
        )
        weights = self.get_importance_weight(mixture_particles.clone().detach())
        surrogate_loss = -mix @ torch.sum(weights * mixture_particles, 1).mean(0)
        loss = -mix @ (mixture_particles.exp().mean(1) + 1e-12).log().mean(0)
        return surrogate_loss, loss.clone().detach().squeeze()

    def get_importance_weight(self, elbo_particles):
        """ Computes the importance weights for the gradients """
        logweights = elbo_particles
        normalized_weights = torch.exp(
            logweights - torch.logsumexp(logweights, -1).unsqueeze(-1)
        )
        return normalized_weights


class ForwardKLOptimizer(DivergenceOptimizer):
    """This learns the variational posterior by minimizing the forward KL divergence
       using importance sampling.

       NOTE: This may suffer in high dimension. Here a clamping strategy may help.
    """

    def __init__(
        self,
        *args,
        proposal="q",
        alpha_decay: float = 0.9,
        is_method="identity",
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.alpha = 1.0
        self.alpha_decay = alpha_decay
        self.proposal = proposal
        self.is_method = is_method
        self._generate_loss_function()
        self._loss_name = "forward_kl"
        self.HYPER_PARAMETERS += ["alpha_decay", "proposal"]
        self.eps = 5e-5

    def _generate_loss_function(self):
        if self.proposal == "q":
            self._loss = self._loss_q_proposal
        elif self.proposal == "prior_q_mix":
            self._loss = self._loss_mix_q
        else:
            raise NotImplementedError("Unknown loss.")

    def weight_f(self, weights):
        if self.is_method == "identity":
            return weights
        elif self.is_method == "clamped":
            return clamp_weights(weights)
        elif self.is_method == "paretto_smoothed":
            return paretto_smoothed_weights(weights)
        else:
            raise NotImplementedError(
                "We only supprot the IS methods 'identity', 'clamped' or 'paretto-smoothed'"
            )

    def effective_sample_size(self, weights):
        """ This is an statistic which indicates how many iid samples from the true
        distribution will give a estimate of comparabe quality """
        M = self.n_particles
        var_mean = 1 + weights.var() / weights.mean() ** 2
        ess = int(M / var_mean) + 1
        M_new = int(self.n_particles * var_mean)
        return ess, M_new

    def _loss_q_proposal(self, x_obs):
        """ This loss use the variational distribution as proposal."""
        samples = self.q.sample((self.n_particles,))
        if hasattr(self.q, "clear_cache"):
            self.q.clear_cache()
        logq = self.q.log_prob(samples)
        logp = self.likelihood.log_prob(x_obs, samples) + self.prior.log_prob(samples)
        with torch.no_grad():
            logweights = logp - logq
            weights = self.weight_f(logweights.exp())
            weights /= weights.sum()

        surrogate = -torch.sum(weights * logq)
        return surrogate, surrogate.detach()

    def _loss_mix_q(self, x_obs):
        """ This loss use a mixture of prior and variational distribution as proposal."""
        k = int(
            torch.binomial(
                torch.tensor([float(self.n_particles)]), torch.tensor([self.alpha])
            )[0]
        )
        sample1 = self.prior.sample((k + 1,))
        sample2 = self.q.sample((self.n_particles - k + 1,))
        samples = torch.vstack([sample1, sample2])
        if hasattr(self.q, "clear_cache"):
            self.q.clear_cache()
        log_q1 = self.prior.log_prob(samples)
        log_q2 = self.q.log_prob(samples)
        log_qs = torch.stack([log_q1, log_q2]).T
        logalphas = torch.tensor([self.alpha, 1 - self.alpha]).log()
        log_proposal = torch.logsumexp(logalphas + log_qs, -1)
        with torch.no_grad():
            logweights = (
                self.likelihood.log_prob(x_obs, samples)
                + self.prior.log_prob(samples)
                - log_proposal
            )
            weights = self.weight_f(logweights.exp())
            weights /= weights.sum()
        surrogate = -torch.sum(weights * log_q2)
        self.alpha = self.alpha * self.alpha_decay
        return surrogate, surrogate.detach()


class RenjeyDivergenceOptimizer(ElboOptimizer):
    r"""This learns the variational posterior by minimizing alpha divergences. For
    alpha=0 we obtain a single sample Monte Carlo estiamte of the IWElbo. For alpha=1 we
    obtain the ELBO. 

    NOTE: We empirically suggest alpha around 0.5

    References: https://arxiv.org/abs/1602.02311
    """

    def __init__(self, *args, alpha=0.5, unbiased=False, **kwargs):
        """
        Args:
            posterior: Variational Posterior object.
            alpha: Fixes which alpha divergence is optimized.
            n_particles: Number of elbo_particels.
            optimizer: PyTorch optimizer class, as default Adam is used.
            clip_value: Max value for gradient clipping.
        """
        self.alpha = alpha
        self.unbiased = unbiased
        super().__init__(*args, **kwargs)
        self._loss_name = "renjey_divergence"
        self.HYPER_PARAMETERS += ["alpha", "unbiased"]
        self.eps = 1e-5

    def _generate_loss_function(self):
        if isinstance(self.alpha, float):
            if self.q.has_rsample:
                if not self.unbiased:
                    self._loss = self.loss_alpha
                else:
                    self._loss = self.loss_alpha_unbiased
            elif isinstance(self.q, Mixture):
                self._loss = self.loss_mixture_alpha
            else:
                raise NotImplementedError()
        elif isinstance(self.alpha, str):
            K = re.findall(r"\d+", self.alpha)
            if "max" in self.alpha:
                if len(K) > 0:
                    self._loss = lambda x: self.loss_max(
                        x, K=min(int(K[0]), self.n_particles)
                    )
                else:
                    self._loss = self.loss_max
            if "min" in self.alpha:
                if len(K) > 0:
                    self._loss = lambda x: self.loss_min(
                        x, K=min(int(K[0]), self.n_particles)
                    )
                else:
                    self._loss = self.loss_max
            if "rand" in self.alpha:
                if len(K) > 0:
                    self._loss = lambda x: self.loss_min(
                        x, K=min(int(K[0]), self.n_particles)
                    )
                else:
                    self._loss = self.loss_max

    def loss_mixture_alpha(self, x_obs):
        """ For mixture distributions with reparameterizable components"""
        mix = torch.softmax(self.q.mixture_distribution.logits, -1)
        mixture_particles = self.generate_mixture_particles(x_obs)
        weights, mean_log_weights = self.get_importance_weight(
            mixture_particles.clone().detach()
        )
        surrogate_loss = -mix @ torch.mean(weights * mixture_particles, 0)
        loss = -mix @ mean_log_weights / (1 - self.alpha)
        return surrogate_loss, loss.clone().detach()

    def loss_alpha_unbiased(self, x_obs):
        """ Unbiased estimate of a surrogate RVB"""
        elbo_particles = self.generate_elbo_particles(x_obs) * (1 - self.alpha)
        _, mean_log_weights = self.get_importance_weight(elbo_particles)
        surrogate_loss = -torch.exp(mean_log_weights)
        loss = -mean_log_weights / (1 - self.alpha)
        return surrogate_loss, loss.clone().detach()

    def loss_alpha(self, x_obs):
        """ Renjy variation bound (RVB)."""
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

    NOTE: This currently does not support mixture distributions.

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
        self._loss_name = "tail_adaptive_fdivergence"
        self.HYPER_PARAMETERS += ["beta"]

        if not self.q.has_rsample:
            raise NotImplementedError(
                "This loss is only implemented for reparameterizable distributions!"
            )

    def _loss(self, x_obs):
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


class FDivergenceOptimizer(ElboOptimizer):
    """ This gives a varaitonal f-divergence bounds for any f-divergence. It requires
    the fenchel convex conjugate f*. 

    NOTE: Most of the bounds will be numerically unstable, care must be taking on the
    choice of f.
    
    References: https://arxiv.org/pdf/2009.13093.pdf
    """

    def __init__(self, *args, name="fKL", f_star=None, f=None, **kwargs):
        """
        Args:
            posterior: Variational Posterior object.
            alpha: Fixes which alpha divergence is optimized.
            n_particles: Number of elbo_particels.
            optimizer: PyTorch optimizer class, as default Adam is used.
            clip_value: Max value for gradient clipping.
        """
        self.name = name
        self.f = f
        self.f_star = f_star
        super().__init__(*args, **kwargs)
        self._loss_name = "fdivergence"
        self.HYPER_PARAMETERS += ["f", "f_star", "name"]
        self.eps = 1e-5

    def _generate_loss_function(self):
        if self.f_star is None:
            if self.name == "fKL":
                self._loss = self._loss_fkl
            elif self.name == "JS":
                self._loss = self._loss_JS
            elif self.name == "TV":
                self._loss = self._loss_TV
            elif self.name == "KL_pol2":
                self._loss = self._loss_KL_pol2
            else:
                pass
        else:
            assert self.f_star is not None, "You must at least specify f_star or a name"
            self._loss = self._loss_general

    def _loss_fkl(self, x_obs):
        """ This requires p(X) > 1/e
        NOTE: The difference to the FKL optimizer is that this minimizes a evidence
        upper bound. Its rather similar but we do not normalize the importance weights.
        As the loss is a expectation we can use the reparameterization trick here which
        we cannot use in the other loss. But we have unnormalized importance weights ..."""
        particles = self.generate_elbo_particles(x_obs)
        weights = particles.exp()
        surrogate_loss = torch.mean(weights * particles)
        return surrogate_loss, surrogate_loss.detach()

    def _loss_TV(self, x_obs):
        """ This is the loss to minimize the total variation"""
        particles = self.generate_elbo_particles(x_obs).exp()
        surrogate_loss = torch.mean(torch.abs(particles - 1))
        return surrogate_loss, surrogate_loss.detach()

    def _loss_KL_pol2(self, x_obs):
        particles = self.generate_elbo_particles(x_obs)
        surrogate_loss = torch.mean(particles ** 2 + particles)
        return surrogate_loss, surrogate_loss.detach()

    def _loss_JS(self, x_obs, domain=1000):
        """ This at most is something proportional to the JS
        
        NOTE To domain should be bounded by log(2)... We extend it to log(100),
        otherwise one would have to clamp all weights..."
        """
        particles = (
            self.generate_elbo_particles(x_obs).exp().clamp_max(np.log(domain) - 1e-5)
        )
        surrogate_loss = torch.mean(-torch.log(domain - torch.exp(particles)))
        return surrogate_loss, surrogate_loss.detach()

    def _loss_general(self, x_obs):
        """ This the general loss, which however is numerically unstable..."""
        particles = self.generate_elbo_particles(x_obs).exp()
        surrogate_loss = self.f_star(particles).mean()
        if self.f is not None:
            loss = self.f(particles).mean()
        else:
            loss = surrogate_loss
        return surrogate_loss, loss.clone().detach()

