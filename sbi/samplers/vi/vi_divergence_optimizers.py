import torch
from torch.distributions import Distribution
from torch import Tensor
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR

from typing import Optional, List, Tuple
from sbi.types import Array


import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy

from sbi.inference.potentials.base_potential import BasePotential


from .vi_utils import (
    filter_kwrags_for_func,
    make_sure_nothing_in_cache,
)

from .vi_sampling import clamp_weights, paretto_smoothed_weights


_VI_method = {}


class DivergenceOptimizer(ABC):
    """This is a wrapper round a PyTorch optimizer and scheduler, which will be used to
    learn the variational distribution. It further contains some methods to evaluate
    convergence and to recover a valid state if something wents wrong.

    This class contains one abstract method '_generate_loss_function' which must be
    implemented und determines the loss function used within variational inference."""

    def __init__(
        self,
        potential_fn: BasePotential,
        q: Distribution,
        n_particles: int = 256,
        clip_value: float = 5.0,
        optimizer: Optional[Optimizer] = Adam,
        scheduler: Optional[object] = ExponentialLR,
        eps: float = 1e-5,
        **kwargs,
    ):
        """This is a wrapper around a PyTorch optimizer which is used to minimize some
         loss for variational inference.

        Args:
            potential_fn: Potential function of the target i.e. the posterior density up
                to normalization constant.
            q: Variational distribution
            n_particles: Number of samples used to estimate gradients.
            clip_value: Norm value on which gradients are clipped.
            optimizer: Base class for an pytorch optimizer.
            scheduler: Base class for an pytorch scheduler.
            eps: This value determines the sensitivity of the convergence checks.
            kwargs: All additional arguments associated with optimizer, scheduler such
                as learning_rate.


        """

        self.potential_fn = potential_fn
        self.q = q
        self.prior = potential_fn.prior

        self.n_particles = n_particles
        self.clip_value = clip_value
        self.device = potential_fn.device
        self.learning_rate = kwargs.get("lr", 1e-3)
        self._kwargs = kwargs

        # This prevents error that would stop optimization.
        self.q.set_default_validate_args(False)
        self.prior.set_default_validate_args(False)

        # Manage modules if present.
        if hasattr(self.q, "modules"):
            self.modules = nn.ModuleList(self.q.modules())
        else:
            self.modules = nn.ModuleList()
        self.modules.train()

        # Ensure that distribution has parameters and that these are on the right device
        if not hasattr(self.q, "parameters"):
            raise ValueError("Your distribution has not parameters please add them!")
        self.to(self.device)

        # Keep a state to resolve invalid values
        self.state_dict = [para.data.clone() for para in self.q.parameters()]

        # Init optimizer and scheduler with correct arguments
        opt_kwargs = filter_kwrags_for_func(optimizer.__init__, kwargs)
        scheduler_kwargs = filter_kwrags_for_func(scheduler.__init__, kwargs)
        scheduler_kwargs["gamma"] = scheduler_kwargs.get("gamma", 1.0)
        self._optimizer = optimizer(self.q.parameters(), **opt_kwargs)
        self._scheduler = scheduler(self._optimizer, **scheduler_kwargs)
        self._scheduler._step_count = 2  # Prevents  warning...

        # Loss and summary
        self.eps = eps
        self.num_step = 0
        self.warm_up_was_done = False
        self.losses = np.ones(2000)
        self.moving_average = np.ones(2000)
        self.moving_std = np.ones(2000)
        self.moving_slope = np.ones(2000)

        # Hyperparameters to change adaptively
        self.HYPER_PARAMETERS = ["n_particles", "clip_value", "eps"]

    @abstractmethod
    def _generate_loss_function(self):
        """This generates the loss function that will be used. Especially it must set
        the '_loss' attribute to an Callable loss function."""
        pass

    def _loss(self, *args, **kwargs):
        """This should be overwritten based on the '_generate_loss_function'. And is
        the function that is evalute as loss."""
        raise NotImplementedError(
            "I should have been overwritten by the '_generate_loss_function'."
        )

    def to(self, device: str):
        """This will move all parameters to the correct device, both for likelihood and
        posterior"""
        self.device = device
        for para in self.q.parameters():
            para.to(device)

    def warm_up(self, num_steps: int, method: str = "prior"):
        """This initializes q, either to follow the prior or the base distribution
        of the flow. This can increase training stability!

        Args:
            num_steps: Number of steps to train.
            method: Method for warmup.

        """
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
        """This updates the current state."""
        for state_para, para in zip(self.state_dict, self.q.parameters()):
            if torch.isfinite(para).all():
                state_para.data = para.data.clone()
            else:
                nn.init.uniform_(para, a=-0.5, b=0.5)

    def resolve_state(self, warm_up_rounds=200):
        """In case the parameters become nan, this method will try to fix the current
        state

        Args:
            warm_up_rounds: Number of warm_up_round one should do after failure.
        """
        for state_para, para in zip(self.state_dict, self.q.parameters()):
            para.data = state_para.data.clone().to(para.device)
        self._optimizer.__init__(self.q.parameters(), self.learning_rate)
        self.warm_up(warm_up_rounds)

    def loss(self, x_obs):
        """Computes the loss function which is optimized.
        Args:
            x_obs: Observed data as input.

        Returns:
            surrogated_loss : The loss that will be differentiated, hence this must be
                differentiable by PyTorch
            loss : This loss will be displayed and used to determine convergence. This
                should not  be differentiable.
        """
        return self._loss(x_obs)

    def step(self, x_obs) -> None:
        """Performs one gradient step

        Args:
            x_obs: Observation which is used.

        """
        self._optimizer.zero_grad()
        surrogate_loss, loss = self.loss(x_obs.to(self.device))
        surrogate_loss.backward()
        if not torch.isfinite(surrogate_loss):
            self.resolve_state()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.clip_value)
        self._optimizer.step()
        self._scheduler.step()
        self.update_loss_stats(loss)
        if (self.num_step % 50) == 0:
            self.update_state()

    def converged(self, considered_values: int = 50) -> bool:
        """Determines convergence based on a estimate of the slope of the loss
        function. If it is smaller than 'eps' then this function will return true.

        Args:
            considered_values: Window over which we will average.

        Returns:
            bool: True if converged, else false.

        """
        if self.num_step < considered_values:
            return False
        else:
            m = self.moving_slope[
                self.num_step - considered_values : self.num_step
            ].mean()
            return abs(m) < self.eps

    def reset_loss_stats(self):
        """This will reset the loss statistics."""
        self.losses = np.ones(2000)
        self.moving_average = np.ones(2000)
        self.moving_std = np.ones(2000)
        self.moving_slope = np.ones(2000)
        self.num_step = 0

    def update_loss_stats(self, loss, window: int = 20) -> None:
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

    def get_loss_stats(self) -> Tuple[Array, Array]:
        """Returns current loss statistics"""
        return (
            self.moving_average[self.num_step - 1],
            self.moving_std[self.num_step - 1],
        )

    def update(self, kwargs: dict):
        """Updates the hyperparameters and scheduler/optimizer kwargs"""
        paras = self.__dict__
        for key, val in kwargs.items():
            if key in self.HYPER_PARAMETERS:
                paras[key] = val

            if key == "self":
                posterior = kwargs[key]
                self.q = posterior.q
                self.potential_fn = posterior.potential_fn
                self.prior = posterior._prior

        if "self" in kwargs:
            kwargs.pop("self")
        opt_kwargs = filter_kwrags_for_func(type(self._optimizer).__init__, kwargs)
        scheduler_kwargs = filter_kwrags_for_func(
            type(self._scheduler).__init__, kwargs
        )
        scheduler_kwargs["gamma"] = scheduler_kwargs.get("gamma", 1.0)
        self._optimizer = type(self._optimizer)(self.q.parameters(), **opt_kwargs)
        self._scheduler = type(self._scheduler)(self._optimizer, **scheduler_kwargs)
        self._scheduler._step_count = 2


def register_VI_method(
    cls: Optional[object] = None,
    name: Optional[str] = None,
):
    """Registers a new VI method, by adding a new Divergence Optimizer class



    Args:
        cls: Class to add
        name: Associated name

    """

    def _register(cls):
        if name is None:
            cls_name = cls.__name__
        else:
            cls_name = name
        if cls_name in _VI_method:
            raise ValueError(f"The VI method {cls_name} is already registered")
        else:
            _VI_method[cls_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_VI_method(name: str) -> DivergenceOptimizer:
    """Returns a specific DivergenceOptimizer using the specified VI method.

    Args:
        name: The name of the method

    Returns:
        DivergenceOptimizer: An divergence optimizer.

    """
    return _VI_method[name]


def get_default_VI_method() -> List[str]:
    return list(_VI_method.keys())


@register_VI_method(name="rKL")
class ElboOptimizer(DivergenceOptimizer):
    r"""This will learn the variational posterior by minimizing the reverse KL
    divergence between q and p i.e. D_KL(q(theta)||p(theta|x_o)), by maximizing the
    ELBO (Evidence Lower Bound).
    """

    def __init__(self, *args, reduce_variance: bool = False, **kwargs):
        """See 'DivergenceOptimizer' for all the base arguments.

        Args:
            reduce_variance: This will reduce the variance of the estimator, especially
                near convergence. Yet for normalizing flows this adds additional cost as
                it requires to evaluate the inverse pass which often can be avoided
                through caching. See [1] for details.

        References:
            [1] Sticking the Landing: Simple, Lower-Variance Gradient Estimators for
                Variational Inference, Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2017,
                https://arxiv.org/abs/1703.09194.

        """
        super().__init__(*args, **kwargs)

        self.reduce_variance = reduce_variance
        make_sure_nothing_in_cache(self.q)
        self._surrogate_q = deepcopy(self.q)

        self._generate_loss_function()
        self.eps = 1e-5
        self.HYPER_PARAMETERS += ["reduce_variance"]

    def _generate_loss_function(self):
        """Generates the loss function depending on the input parameters."""
        if self.q.has_rsample:
            self._loss = self.loss_rsample
        else:
            raise NotImplementedError(
                "Currently only reparameterizable distributions are supported."
            )

    def loss_rsample(self, x_obs):
        """Computes the ELBO"""
        elbo_particles = self.generate_elbo_particles(x_obs)
        loss = -elbo_particles.mean()
        return loss, loss.clone().detach()

    def generate_elbo_particles(self, x_obs, num_samples=None):
        """Generates individual ELBO particles i.e. logp(theta, x_o) - logq(theta)."""
        if num_samples is None:
            num_samples = self.n_particles
        samples = self.q.rsample((num_samples,))
        if self.reduce_variance:
            self.update_surrogate_q()
            log_q = self._surrogate_q.log_prob(samples)
        else:
            log_q = self.q.log_prob(samples)
        self.potential_fn.x_o = x_obs
        log_potential = self.potential_fn(samples)
        elbo = log_potential - log_q
        return elbo

    def update_surrogate_q(self):
        """Updates the surrogate with new parameters."""
        for param, param_surro in zip(
            self.q.parameters(), self._surrogate_q.parameters()
        ):
            param_surro.data = param.data
            param_surro.requires_grad = False


@register_VI_method(name="IW")
class IWElboOptimizer(ElboOptimizer):
    r"""This learns the variational posterior by minimizing the importance weighted
    ELBO, which is an tighter bound to the evidence but also promotes a support covering
    behaviour.

    NOTE: You may want to turn on 'reduce_variance' here as this loss leads to gradient
    estimates with vanishing signal to noise ratio. This is relevant for large K,
    especially K > n_particles.

    NOTE: Technically this does not minimize a valid divergence between q and p, yet it
    does optimizer q as proposal for sampling importance resampling.

    References:
        Importance Weighted Autoencoders, Yuri Burda, Roger Grosse, Ruslan
        Salakhutdinov, 2016, https://arxiv.org/abs/1509.00519.
    """

    def __init__(self, *args, K=8, dreg=False, **kwargs):
        """See 'DivergenceOptimizer' for all the base arguments.



        Args:
            K: Number of samples used within estimating a single gradient. In total
                n_particles x K samples are used.
            dreg: Doubly reparmeterized gradient estimator as proposed in [1]. It is
                based on the 'reduced_variance' already present, but leads to an
                unbiased estimate on this objective in contrast.

        References:
            [1] _Doubly Reparameterized Gradient Estimators for Monte Carlo Objectives_,
                George Tucker, Dieterich Lawson, Shixiang Gu, Chris J. Maddison, 2018,
                https://arxiv.org/abs/1810.04152.


        """
        super().__init__(*args, **kwargs)
        self.K = K
        self.loss_name = "iwelbo"
        self.eps = 1e-7
        self.dreg = dreg
        self.HYPER_PARAMETERS += ["K", "dreg"]
        if dreg:
            self.reduce_variance = True

    def loss_rsample(self, x_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the IWELBO loss.

        Args:
            x_obs: Observation

        Returns:
            surrogate_loss: Loss to differentiate.
            loss: Loss to diplay.

        """
        elbo_particles = self.generate_elbo_particles(x_obs, self.n_particles * self.K)
        elbo_particles = elbo_particles.reshape(self.n_particles, self.K)
        weights = self.get_importance_weight(elbo_particles.clone().detach())
        surrogate_loss = -(weights * elbo_particles).sum(-1).mean(0)
        loss = -torch.mean(torch.exp(elbo_particles) + 1e-20, -1).log().mean()
        return surrogate_loss, loss.clone().detach()

    def get_importance_weight(self, elbo_particles: Tensor) -> Tensor:
        """Generates importance weights in a numerically stable fashion.

        Args:
            elbo_particles: ELements contain log potential - log q.

        Returns:
            Tensor: Normalized importance weights.

        """
        logweights = elbo_particles
        normalized_weights = torch.exp(
            logweights - torch.logsumexp(logweights, -1).unsqueeze(-1)
        )
        if self.dreg:
            normalized_weights = normalized_weights ** 2
        return normalized_weights


@register_VI_method(name="fKL")
class ForwardKLOptimizer(DivergenceOptimizer):
    """This learns the variational posterior by minimizeing the forward KL divergence
    i.e. D_KL(p(theta|x_o)|| q(theta)). The typically necessary samples from the
    posterior are not required by using importance sampling tenchniques.

    NOTE: Whereas in the previous cases n_particles mainly reduces the variance, in this
    case it also decrease the bias. Using n_particles=1 is not usefull.

    References:
        [1] _Variational Refinement for Importance Sampling Using the Forward
            Kullback-Leibler Divergence_, Ghassen Jerfel, Serena Wang, Clara Fannjiang,
            Katherine A. Heller, Yian Ma, Michael I. Jordan, 2021,
            https://arxiv.org/abs/2106.15980.
    """

    def __init__(
        self,
        *args,
        proposal: str = "q",
        is_method: str = "identity",
        **kwargs,
    ):
        """See 'DivergenceOptimizer' for base arguments.

        Args:
            proposal: The proposal used we currently support only ['q']
            is_method: Importance sampling method we currently support ['identity',
                'clamped', 'paretto_smoothed'].


        """
        super().__init__(*args, **kwargs)

        self.alpha = 1.0
        self.proposal = proposal
        self.is_method = is_method
        self._generate_loss_function()
        self._loss_name = "forward_kl"
        self.HYPER_PARAMETERS += ["is_method", "proposal"]
        self.eps = 5e-5

    def _generate_loss_function(self):
        """This generates the loss function."""
        if self.proposal == "q":
            self._loss = self._loss_q_proposal
        else:
            raise NotImplementedError("Unknown loss.")

    def weight_f(self, weights: Tensor) -> Tensor:
        """This applies a weight transform to the importance weights.

        Args:
            weights: Importance weights to process.

        Returns:
            Tensor : Processed importance weights.

        """
        if self.is_method == "identity":
            return weights
        elif self.is_method == "clamped":
            return clamp_weights(weights)
        elif self.is_method == "paretto_smoothed":
            return paretto_smoothed_weights(weights)
        else:
            raise NotImplementedError(
                "We only supprot the IS methods 'identity', 'clamped'\
                or 'paretto-smoothed'"
            )

    def _loss_q_proposal(self, x_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """This gives an importance sampling estimate of the forward KL divergence.

        Args:
            x_obs: Obsevation.

        Returns:
            Tuple[Tensor, Tensor]: Surrogate loss to differentiate and to display.

        """
        samples = self.q.sample((self.n_particles,))
        if hasattr(self.q, "clear_cache"):
            self.q.clear_cache()
        logq = self.q.log_prob(samples)
        self.potential_fn.x_o = x_obs
        logp = self.potential_fn(samples)
        with torch.no_grad():
            logweights = logp - logq
            weights = self.weight_f(logweights.exp())
            weights /= weights.sum()

        surrogate = -torch.sum(weights * logq)
        loss = torch.sum(weights * (logp - logq))
        return surrogate, loss.detach()


@register_VI_method(name="alpha")
class RenyiDivergenceOptimizer(ElboOptimizer):
    r"""This learns the variational posterior by minimizing Renyi alpha divergences. For
    alpha=0 we obtain this is equivalent to the IWELBO with n_particles=1 and
    K=n_particles. For alpha=1 this is equivalent to the 'ElboOptimizer'.

    NOTE: For alpha < 1 the divergence is more mass covering, for alpha > 1 the
    divergence is more mode seeking.

    NOTE: For small alpha, you i.e. alpha=0.1 you may require to
    turn on reduce_variance or dreg.

    References:
        [1] _Rényi Divergence Variational Inference_, Yingzhen Li, Richard E. Turner,
            2016,https://arxiv.org/abs/1602.02311.
    """

    def __init__(self, *args, alpha=0.5, unbiased=False, dreg=False, **kwargs):
        """See 'ElboOptimizer' for the base arguments.

        Args:
            alpha: Determines the alpha divergence to which is used. For alpha < 1 the
                divergence is more mass covering, for alpha > 1 the divergence is more
                mode seeking.
            unbiased: We use the biased bound as proposed in [1], but one can also use
                unbiased one.
            dreg: Doubly reparmeterized gradient estimator as proposed in [2]. It is
                based on the 'reduced_variance' already present, but leads to an
                unbiased estimate on this objective in contrast.

        Reference:
            [1] _Rényi Divergence Variational Inference_, Yingzhen Li, Richard E.
                Turner, 2016,https://arxiv.org/abs/1602.02311.
            [2] _Doubly Reparameterized Gradient Estimators for Monte Carlo Objectives_,
                George Tucker, Dieterich Lawson, Shixiang Gu, Chris J. Maddison, 2018,
                https://arxiv.org/abs/1810.04152.

        """
        self.alpha = alpha
        self.unbiased = unbiased
        super().__init__(*args, **kwargs)
        self.HYPER_PARAMETERS += ["alpha", "unbiased", "dreg"]
        self.eps = 1e-5
        self.dreg = dreg
        if dreg:
            self.reduce_variance = True

    def _generate_loss_function(self):
        if isinstance(self.alpha, float):
            if self.q.has_rsample:
                if not self.unbiased:
                    self._loss = self.loss_alpha
                else:
                    self._loss = self.loss_alpha_unbiased
            else:
                raise NotImplementedError(
                    "Currently we only support reparameterizable distributions"
                )

    def loss_alpha_unbiased(self, x_obs):
        """Unbiased estimate of a surrogate RVB"""
        elbo_particles = self.generate_elbo_particles(x_obs) * (1 - self.alpha)
        _, mean_log_weights = self.get_importance_weight(elbo_particles)
        surrogate_loss = -torch.exp(mean_log_weights)
        loss = -mean_log_weights / (1 - self.alpha)
        return surrogate_loss, loss.clone().detach()

    def loss_alpha(self, x_obs):
        """Renjy variation bound (RVB)."""
        elbo_particles = self.generate_elbo_particles(x_obs)
        weights, mean_log_weights = self.get_importance_weight(
            elbo_particles.clone().detach()
        )
        surrogate_loss = -torch.mean(weights * elbo_particles)
        loss = -mean_log_weights / (1 - self.alpha)
        return surrogate_loss, loss

    def get_importance_weight(self, elbo_particles):
        """Computes the importance weights for the gradients"""
        logweights = (1 - self.alpha) * elbo_particles
        mean_log_weights = torch.logsumexp(logweights, 0) - np.log(self.n_particles)
        normed_logweights = logweights - mean_log_weights
        weights = normed_logweights.exp()
        if self.dreg:
            weights = weights ** 2
        return weights, mean_log_weights
