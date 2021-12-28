from typing import Callable, Dict, Optional, Tuple, Union
import torch
from torch import Tensor, optim
import torch.distributions.transforms as torch_tf


def get_maximum(
    potential_fn: Callable,
    inits: Tensor,
    potential_tf: Optional[torch_tf.Transform] = None,
    num_iter: int = 1_000,
    num_to_optimize: int = 100,
    learning_rate: float = 0.01,
    save_best_every: int = 10,
    show_progress_bars: bool = False,
    interruption_note: str = "",
) -> Tuple[Tensor, Tensor]:
    """
    Returns the `argmax` and `max` of a `potential_fn`.

    The method can be interrupted (Ctrl-C) when the user sees that the log-probability
    converges. The best estimate will be returned.

    The maximum is obtained by running gradient ascent from given starting parameters.
    After the optimization is done, we select the parameter set that has the highest
    `potential_fn` value after the optimization.

    Warning: The default values used by this function are not well-tested. They might
    require hand-tuning for the problem at hand.

    Args:
        potential_fn: The function on which to optimize.
        inits: The initial parameters at which to start the gradient ascent steps.
        dist_specifying_bounds: Distribution the specifies bounds for the optimization.
            If it is a `sbi.utils.BoxUniform`, we transform the space into
            unconstrained space and carry out the optimization there.
        num_iter: Number of optimization steps that the algorithm takes
            to find the MAP.
        num_to_optimize: From the drawn `num_init_samples`, use the `num_to_optimize`
            with highest log-probability as the initial points for the optimization.
        learning_rate: Learning rate of the optimizer.
        save_best_every: The best log-probability is computed, saved in the
            `map`-attribute, and printed every `save_best_every`-th iteration.
            Computing the best log-probability creates a significant overhead (thus,
            the default is `10`.)
        show_progress_bars: Whether or not to show a progressbar for the optimization.
        interruption_note: The message printed when the user interrupts the
            optimization.

    Returns:
        The `argmax` and `max` of the `potential_fn`.
    """

    if potential_tf is None:
        potential_tf = torch_tf.IndependentTransform(
            torch_tf.identity_transform, reinterpreted_batch_ndims=1
        )
    else:
        potential_tf = potential_tf

    init_probs = potential_fn(inits).detach()

    # Pick the `num_to_optimize` best init locations.
    sort_indices = torch.argsort(init_probs, dim=0)
    sorted_inits = inits[sort_indices]
    optimize_inits = sorted_inits[-num_to_optimize:]

    # The `_overall` variables store data accross the iterations, whereas the
    # `_iter` variables contain data exclusively extracted from the current
    # iteration.
    best_log_prob_iter = torch.max(init_probs)
    best_theta_iter = sorted_inits[-1]
    best_theta_overall = best_theta_iter.detach().clone()
    best_log_prob_overall = best_log_prob_iter.detach().clone()

    argmax_ = best_theta_overall
    max_val = best_log_prob_overall

    optimize_inits = potential_tf(optimize_inits)
    optimize_inits.requires_grad_(True)
    optimizer = optim.Adam([optimize_inits], lr=learning_rate)

    iter_ = 0

    # Try-except block in case the user interrupts the program and wants to fall
    # back on the last saved `.map_`. We want to avoid a long error-message here.
    try:

        while iter_ < num_iter:

            optimizer.zero_grad()
            probs = potential_fn(potential_tf.inv(optimize_inits)).squeeze()
            loss = -probs.sum()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if iter_ % save_best_every == 0 or iter_ == num_iter - 1:
                    # Evaluate the optimized locations and pick the best one.
                    log_probs_of_optimized = potential_fn(
                        potential_tf.inv(optimize_inits)
                    )
                    best_theta_iter = optimize_inits[
                        torch.argmax(log_probs_of_optimized)
                    ]
                    best_log_prob_iter = potential_fn(potential_tf.inv(best_theta_iter))
                    if best_log_prob_iter > best_log_prob_overall:
                        best_theta_overall = best_theta_iter.detach().clone()
                        best_log_prob_overall = best_log_prob_iter.detach().clone()

                if show_progress_bars:
                    print(
                        f"""Optimizing MAP estimate. Iterations: {iter_+1} /
                        {num_iter}. Performance in iteration
                        {divmod(iter_+1, save_best_every)[0] * save_best_every}:
                        {best_log_prob_iter.item():.2f} (= unnormalized log-prob""",
                        end="\r",
                    )
                argmax_ = potential_tf.inv(best_theta_overall)
                max_val = best_log_prob_overall

            iter_ += 1

    except KeyboardInterrupt:
        interruption = f"Optimization was interrupted after {iter_} iterations. "
        print(interruption + interruption_note)
        return argmax_, max_val

    return potential_tf.inv(best_theta_overall), max_val
