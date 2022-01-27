import torch
from sbi.utils import gradient_ascent

from sbi.types import Shape
from torch import Tensor
from typing import List, Optional, Union, Callable


class NeuralPosteriorEnsemble:
    r"""Wrapper class to bundle together different posterior instances into an ensemble.

    This class creates a posterior ensemble from a set of N different, already trained
    posterior estimators $p_{i}(\theta|x_o)$, where $i \in \{i,...,N\}$.

    It can wrap all posterior classes available in sbi and even a mixture of different
    posteriors, i.e. obtained via SNLE and SNPE at the same time, since it only
    provides a pass-through to the class-methods of each posterior in the ensemble.

    So far `log_prob()`, `sample()` and `map()` functionality are supported.

    Example:

    ```
    import torch
    from joblib import Parallel, delayed
    from sbi.examples.minimal import simple

    ensemble_size = 10
    posteriors = Parallel(n_jobs=-1)(delayed(simple)() for i in range(ensemble_size))

    ensemble = NeuralPosteriorEnsemble(posteriors)
    ensemble.set_default_x(torch.zeros((3,)))
    ensemble.sample((1,))
    ```

    Attributes:
        posteriors: List of the posterior estimators making up the ensemble.
        num_components: Number of posterior estimators.
        weights: Weight of each posterior distribution. If none are provided each
            posterior is weighted with 1/N.
        default_x: Used in `.sample(), .log_prob()` as default conditioning context.
    """

    def __init__(self, posteriors: List, weights: Optional[List] or Tensor = None):
        """

        Args:
            posteriors: List containing the trained posterior instances that will make
                up the ensemble.
            weights: Assign weights to posteriors manually, otherwise they will be
                weighted with 1/N.
        """
        self.posteriors = posteriors
        self.num_components = len(posteriors)
        self.weights = weights
        self.default_x = None

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: List[float] or Tensor):
        """Set relative weight for each posterior in the ensemble.

        Weights are normalised.

        Args:
            weights: Assignes weight to each posterior distribution.
        """
        if weights is None:
            self._weights = torch.tensor(
                [1.0 / self.num_components for _ in range(self.num_components)]
            )
        elif type(weights) == Tensor or type(weights) == List:
            self._weights = torch.tensor(weights) / sum(weights)
        else:
            raise TypeError

    def sample(self, sample_shape: Shape = torch.Size(), **kwargs) -> Tensor:
        r"""Return samples from posterior ensemble.

        All kwargs are passed directly through to `posterior.sample()`.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior
                ensemble. If sample_shape is multidimensional we simply draw
                    `sample_shape.
            numel()` samples and then reshape into the desired shape.
        """
        num_samples = torch.Size(sample_shape).numel()
        idxs = torch.multinomial(self._weights, num_samples, replacement=True)
        samples = []
        for c, n in torch.vstack(idxs.unique(return_counts=True)).T:
            sample_shape_c = torch.Size((n,))
            samples.append(self.posteriors[c].sample(sample_shape_c, **kwargs))
        return torch.vstack(samples).reshape(*sample_shape, -1)

    def log_prob(self, theta: Tensor, individually: bool = False, **kwargs) -> Tensor:
        r"""Returns the average log-probability of the posterior ensemble

        $\sum_{i}^{N} w_{i} p_i(\theta|x)$.

        All kwargs are passed directly through to `posterior.log_prob()`.

        Args:
            theta: Parameters $\theta$.
            individually: If true, returns log weights and log_probs individually.

        Returns:
            `(len(θ),)`-shaped average log posterior probability $\log p(\theta|x)$ for
            θ in the support of the prior, -∞ (corresponding to 0 probability) outside.
        """
        log_probs = torch.stack(
            [posterior.log_prob(theta, **kwargs) for posterior in self.posteriors]
        )
        log_weights = torch.log(self._weights).reshape(-1, 1)

        if individually:
            return log_weights, log_probs
        else:
            return torch.logsumexp(log_weights.expand_as(log_probs) + log_probs, dim=0)

    def set_default_x(self, x: Tensor) -> "NeuralPosteriorEnsemble":
        """Set new default x for `.sample(), .log_prob()` to use as conditioning context.

        This is a pure convenience to avoid having to repeatedly specify `x` in calls to
        `.sample()` and `.log_prob()` - only θ needs to be passed.

        This convenience is particularly useful when the posterior ensemble is focused,
        i.e. has been trained over multiple rounds to be accurate in the vicinity of a
        particular `x=x_o` (you can check if your posterior object is focused by
        printing one exemplary component of the ensemble).

        NOTE: this method is chainable, i.e. will return the NeuralPosteriorEnsemble
        object so that calls like `posterior_enemble.set_default_x(my_x).sample(mytheta)
        ` are possible.

        Args:
            x: The default observation to set for every posterior $p_i(theta|x)$ in the
            ensemble.
        Returns:
            `NeuralPosteriorEnsemble` that will use a default `x` when not explicitly
            passed.
        """
        self.default_x = x
        for posterior in self.posteriors:
            posterior.set_default_x(x)
        return self

    def ensemble_potential_fn_init(self, x: Optional[Tensor] = None) -> Callable:
        """Initialises the ensemble potential function.

        The potential functions of every component are combined to produce an
        ensemble potential.

        Args:
            x: sets context. If None is provided, the default_x will be used.

        Returns:
            ensemble_potential_fn: Differentiable potential function of the entire
            ensemble.
        """
        potential_fns = []
        for posterior in self.posteriors:
            potential_fn = posterior.potential_fn
            potential_fn.set_x(posterior._x_else_default_x(x))
            potential_fns.append(potential_fn)

        def ensemble_potential_fn(theta: Tensor) -> Callable:
            log_probs = [fn(theta) for fn in potential_fns]
            log_probs = torch.vstack(log_probs)
            potential_fn = torch.logsumexp(
                torch.log(self._weights.reshape(-1, 1)).expand_as(log_probs)
                + log_probs,
                dim=0,
            )
            return potential_fn

        return ensemble_potential_fn

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
        individually: bool = False,
    ) -> Tensor:
        r"""Returns the average maximum-a-posteriori estimate (MAP).

        Computes MAP estimate across the whole ensemble or for each component
        individually. All args and kwargs are passed directly through to
        `gradient_ascent`.

        The routine can be interrupted (individually) with [Ctrl-C], when the user sees
        that the log-probability converges. The best estimate will be saved in `self.
        posteriors[idx].map_`.

        For more details of how the MAP estimate is obtained see `.map()` docstring of
        self.posteriors[idx].

        Args:
            x: Observed data at which to evaluate the MAP.
            num_iter: Number of optimization steps that the algorithm takes
                to find the MAP.
            num_to_optimize: From the drawn `num_init_samples`, use the
                `num_to_optimize` with highest log-probability as the initial points
            learning_rate: Learning rate of the optimizer.
            init_method: How to select the starting parameters for the optimization. If
                it is a string, it can be either [`posterior`, `prior`], which samples
                the respective distribution `num_init_samples` times. If it is a
                tensor, the tensor will be used as init locations.
            num_init_samples: Draw this number of samples from the posterior and
                evaluate the log-probability of all of them.
                for the optimization.
            save_best_every: The best log-probability is computed, saved in the
                `map`-attribute, and printed every `save_best_every`-th iteration.
                Computing the best log-probability creates a significant overhead
                (thus, the default is `10`.)
            show_progress_bars: Whether or not to show a progressbar for sampling from
                the posterior.
            individually: If true, returns log weights and MAPs individually.

        Returns:
            The ensemble MAP estimate or individual log_weigths and component MAP
            estimate if individually == True.
        """

        if individually:
            maps = []
            log_weights = torch.log(self._weights).reshape(-1, 1)

            for posterior in self.posteriors:
                maps.append(
                    posterior.map(
                        x=x,
                        num_iter=num_iter,
                        num_to_optimize=num_to_optimize,
                        learning_rate=learning_rate,
                        init_method=init_method,
                        num_init_samples=num_init_samples,
                        save_best_every=save_best_every,
                        show_progress_bars=show_progress_bars,
                    )
                )
            maps = torch.stack(maps)
            return log_weights, maps

        else:
            potential_fn = self.ensemble_potential_fn_init(x)

            if init_method == "posterior":
                inits = self.sample((num_init_samples,))
            # elif init_method == "proposal":
            #     inits = self.proposal.sample((num_init_samples,))
            elif isinstance(init_method, Tensor):
                inits = init_method
            else:
                raise ValueError

            return gradient_ascent(
                potential_fn=potential_fn,
                inits=inits,
                # theta_transform=self.theta_transform,
                num_iter=num_iter,
                num_to_optimize=num_to_optimize,
                learning_rate=learning_rate,
                save_best_every=save_best_every,
                show_progress_bars=show_progress_bars,
            )[0]
