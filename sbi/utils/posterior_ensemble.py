import torch
from typing import List, Optional
from torch import Tensor
from sbi.types import Shape


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

    Args:
        posteriors: List containing the trained posterior instances that will make up
        the ensemble.
        weights: Assign weights to posteriors manually, otherwise they will be
        weighted with 1/N.

    Attributes:
        posteriors: List of the posterior estimators making up the ensemble.
        num_components: Number of posterior estimators.
        weights: Weight of each posterior distribution. If none are provided each
            posterior is weighted with 1/N.
    """

    def __init__(self, posteriors: List, weights: Optional[List] = None):
        self.posteriors = posteriors
        self.num_components = len(posteriors)
        self.weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: List[float]):
        """Set relative weight for each posterior in the ensemble.

        Weights are normalised.

        Args:
            weights: Assignes weight to each posterior distribution.
        """
        if type(weights) == list:
            weights = torch.tensor(weights) / sum(weights)
            self._weights = weights
        else:
            self._weights = torch.tensor(
                [1.0 / self.num_components for _ in range(self.num_components)]
            )

    def sample(self, sample_shape: Shape = torch.Size(), **kwargs) -> Tensor:
        r"""Return samples from posterior ensemble.

        All kwargs are passed directly through to `posterior.sample()`.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior
            ensemble. If sample_shape is multidimensional we simply draw `sample_shape.
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

    def set_default_x(self, x: Tensor) -> "NeuralPosteriorEnemble":
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
        for posterior in self.posteriors:
            posterior.set_default_x(x)
        return self

    def map(self, individually: bool = False, **kwargs) -> Tensor:
        r"""Returns the average maximum-a-posteriori estimate (MAP).

        Computes MAP estimate for each posterior in the ensemble individually and
        averages them. All args and kwargs are passed directly through to
        `posterior.map()`.

        Since `posterior.map()` is computed one by one, the subroutines of each MAP
        estimate can be interrupted individually (Ctrl-C), when the user sees that the
        log-probability converges. The best estimate will be saved in `self.posteriors
        [idx].map_` and added to the average MAP estimate.

        For more details of how the MAP estimate is obtained see `.map()` docstring of
        self.posteriors[idx].

        Args:
            individually: If true, returns log weights and MAPs individually.

        Kwargs:
            x: Observed data at which to evaluate the MAP.
            num_iter: Number of optimization steps that the algorithm takes
                to find the MAP.
            learning_rate: Learning rate of the optimizer.
            init_method: How to select the starting parameters for the optimization. If
                it is a string, it can be either [`posterior`, `prior`], which samples
                the respective distribution `num_init_samples` times. If it is a
                tensor, the tensor will be used as init locations.
            num_init_samples: Draw this number of samples from the posterior and
                evaluate the log-probability of all of them.
            num_to_optimize: From the drawn `num_init_samples`, use the
                `num_to_optimize` with highest log-probability as the initial points
                for the optimization.
            save_best_every: The best log-probability is computed, saved in the
                `map`-attribute, and printed every `save_best_every`-th iteration.
                Computing the best log-probability creates a significant overhead
                (thus, the default is `10`.)
            show_progress_bars: Whether or not to show a progressbar for sampling from
                the posterior.
            log_prob_kwargs: Will be empty for SNLE and SNRE. Will contain
                {'norm_posterior': True} for SNPE.

        Returns:
            The average MAP estimate.
        """
        maps = []
        log_weights = torch.log(self._weights).reshape(-1, 1)

        for posterior in self.posteriors:
            maps.append(posterior.map(**kwargs))
        maps = torch.stack(maps)

        if individually:
            return log_weights, maps
        else:
            return torch.logsumexp(log_weights + maps, dim=0)
