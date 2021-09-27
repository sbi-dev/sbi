from .mixture_of_flows import build_mixture
from .flows import build_flow
from .divergence_optimizers import (
    ElboOptimizer,
    IWElboOptimizer,
    RenjeyDivergenceOptimizer,
    TailAdaptivefDivergenceOptimizer,
    ForwardKLOptimizer,
    FDivergenceOptimizer,
    make_sure_nothing_in_cache,
)
from sbi.utils.torchutils import (
    atleast_2d_float32_tensor,
    Array,
    Tensor,
    ensure_x_batched,
)
from sbi.vi.sampling import paretto_smoothed_weights, clamp_weights

import torch
from warnings import warn
from typing import Optional, Callable
from tqdm import tqdm

import numpy as np


# Some of the main arguments for
KWARGS_Q = ["flow", "num_flow", "num_components", "rsample", "permute", "batch_norm"]


def expectation(
    posterior,
    f: Callable,
    x: Optional[Tensor] = None,
    method: str = "naive",
    num_samples: int = 10000,
):
    """Computes the expectation with respect to the posterior E_q[f(X)]

        Args:
        f: Function for which we will compute the expectation
        method: Method either naive (just using the variatioanl posterior), is
        (importance sampling) or psis (pareto smoothed importance sampling).
    """
    # TODO use vi_parameter attribute...
    samples = posterior.sample((num_samples,))
    if method == "naive":
        return f(samples).mean(0)
    else:
        with torch.no_grad():
            x_obs = atleast_2d_float32_tensor(posterior._x_else_default_x(x))
            x_obs = ensure_x_batched(x_obs)
            obs = x_obs.repeat(num_samples, 1)
            logweights = (
                posterior.net.log_prob(obs, samples)
                + posterior._prior.log_prob(samples)
                - posterior._q.log_prob(samples)
            )
            weights = torch.exp(logweights)
        if method == "is":
            pass
        elif method == "psis":
            weights = paretto_smoothed_weights(weights)
        elif method == "clamped":
            weights = clamp_weights(weights)
        else:
            raise NotImplementedError(
                "We only have the methods naive, is, psis and clamped."
            )
        weights /= weights.sum()
        return torch.sum(f(samples) * weights.unsqueeze(-1), 0)


def train_posterior(
    posterior,
    x: Optional[Array] = None,
    loss: str = "elbo",
    n_particles: Optional[int] = 128,
    learning_rate: float = 1e-3,
    gamma: float = 0.999,
    max_num_iters: Optional[int] = 2000,
    min_num_iters: Optional[int] = 10,
    clip_value: Optional[float] = 5.0,
    warm_up_rounds: int = 100,
    retrain_from_scratch: bool = False,
    reset_optimizer: bool = False,
    show_progress_bar: bool = True,
    check_for_convergence: bool = True,
    **kwargs,
):
    """This methods trains the variational posterior.
        
        Args:
            x: The observation
            loss: The loss that is minimimzed, default is the ELBO
            n_particles: Number of samples to approximate expectations.
            learning_rate: Learning rate of the optimizer
            gamma: Learning rate decay per iteration
            max_num_iters: Maximum number of iterations
            clip_value: Gradient clipping value
            warm_up_rounds: Initialize the posterior as the prior.
            retrain_from_scratch: Retrain the flow
            resume_training: Resume training the flow
            show_progress_bar: Show the progress bar
        """

    # Init q and the optimizer if necessary
    if retrain_from_scratch:
        posterior._q = build_q(
            posterior._prior.event_shape,
            posterior._prior.support,
            **posterior.vi_parameters,
        )
        posterior._optimizer = build_optimizer(
            posterior,
            loss,
            lr=learning_rate,
            clip_value=clip_value,
            gamma=gamma,
            n_particles=n_particles,
            **kwargs,
        )

    if reset_optimizer or posterior._optimizer._loss_name != loss:
        posterior._optimizer = build_optimizer(
            posterior,
            loss,
            lr=learning_rate,
            clip_value=clip_value,
            gamma=gamma,
            n_particles=n_particles,
            **kwargs,
        )
        posterior._loss = loss

    # Check context
    x = atleast_2d_float32_tensor(posterior._x_else_default_x(x)).to(posterior._device)
    if not posterior._allow_iid_x:
        posterior._ensure_single_x(x)
    posterior._ensure_x_consistent_with_x_shape(x)

    # Optimize
    posterior._optimizer.update({**locals(), **kwargs})
    optimizer = posterior._optimizer
    optimizer.reset_loss_stats()

    if show_progress_bar:
        iters = tqdm(range(max_num_iters))
    else:
        iters = range(max_num_iters)

    # Warmup before training
    if not optimizer.warm_up_was_done:
        if show_progress_bar:
            iters.set_description("Warmup phase, this takes some seconds...")
        optimizer.warm_up(warm_up_rounds)

    for i in iters:
        optimizer.step(x)
        mean_loss, std_loss = optimizer.get_loss_stats()
        # Update progress bar
        if show_progress_bar:
            iters.set_description(
                f"Loss: {np.round(mean_loss, 2)} Std: {np.round(std_loss, 2)}"
            )
        # Check for convergence
        if check_for_convergence and i > min_num_iters:
            if optimizer.converged():
                if show_progress_bar:
                    print(f"\nConverged with loss: {np.round(mean_loss, 2)}")
                break
    if show_progress_bar:
        try:
            k = round(float(optimizer.evaluate(x)), 3)
            print(f"Quality Score: {k} (smaller values are good, should be below 1)")
            if k > 1:
                warn(
                    "The quality of the variational posterior seems to be bad, increase the training iterations or consider a different variational family!"
                )
        except:
            posterior._q = build_q(
                posterior._prior.event_shape,
                posterior._prior.support,
                **posterior.vi_parameters,
            )
            posterior._optimizer.q = posterior._q


def build_q(
    event_shape: torch.Size,
    support: torch.distributions.constraints.Constraint,
    flow: str = "spline_autoregressive",
    num_components: int = 1,
    **kwargs,
):
    """This method builds an normalizing flow or a mixture of normalizing flows.
    Args:
        event_shape: Event shape
        support: The support of the distribtuion
        flow: The type of flow
        num_components: Number of mixture components, default is one.
    
    Returns:
        [type]: [description]
    
    """
    if num_components > 1:
        return build_mixture(
            event_shape, support, num_components=num_components, type=flow, **kwargs
        )
    else:
        return build_flow(event_shape, support, type=flow, **kwargs)


def build_optimizer(posterior, loss, **kwargs):
    """ This methods builds an optimizer"""
    if loss.lower() == "elbo":
        optimizer = ElboOptimizer(posterior, **kwargs)
    elif loss.lower() == "iwelbo":
        optimizer = IWElboOptimizer(posterior, **kwargs)
    elif loss.lower() == "renjey_divergence":
        optimizer = RenjeyDivergenceOptimizer(posterior, **kwargs)
    elif loss.lower() == "tail_adaptive_fdivergence":
        optimizer = TailAdaptivefDivergenceOptimizer(posterior, **kwargs)
    elif loss.lower() == "forward_kl":
        optimizer = ForwardKLOptimizer(posterior, **kwargs)
    elif loss.lower() == "fdivergence":
        optimizer = FDivergenceOptimizer(posterior, **kwargs)
    else:
        raise NotImplementedError("Unknown loss...")
    return optimizer
