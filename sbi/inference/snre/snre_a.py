from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor, nn, ones

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.snre.snre_base import RatioEstimator
from sbi.types import TensorboardSummaryWriter
from sbi.utils import del_entries


class SNRE_A(RatioEstimator):
    def __init__(
        self,
        simulator: Callable,
        prior,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        classifier: Union[str, Callable] = "resnet",
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        logging_level: Union[int, str] = "warning",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
        show_round_summary: bool = False,
    ):
        r"""AALR[1], here known as SNRE_A.

        [1] _Likelihood-free MCMC with Amortized Approximate Likelihood Ratios_, Hermans
            et al., ICML 2020, https://arxiv.org/abs/1903.04057

        Args:
            simulator: A function that takes parameters $\theta$ and maps them to
                simulations, or observations, `x`, $\mathrm{sim}(\theta)\to x$. Any
                regular Python callable (i.e. function or class with `__call__` method)
                can be used.
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            num_workers: Number of parallel workers to use for simulations.
            simulation_batch_size: Number of parameter sets that the simulator
                maps to data x at once. If None, we simulate all parameter sets at the
                same time. If >= 1, the simulator has to process data of shape
                (simulation_batch_size, parameter_dimension).
            classifier: Classifier trained to approximate likelihood ratios. If it is
                a string, use a pre-configured network of the provided type (one of
                linear, mlp, resnet). Alternatively, a function that builds a custom
                neural network can be provided. The function will be called with the
                first batch of simulations (theta, x), which can thus be used for shape
                inference and potentially for z-scoring. It needs to return a PyTorch
                `nn.Module` implementing the classifier.
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`, `hmc`, `nuts`.
                Currently defaults to `slice_np` for a custom numpy implementation of
                slice sampling; select `hmc`, `nuts` or `slice` for Pyro-based sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains, `init_strategy`
                for the initialisation strategy for chains; `prior` will draw init
                locations from prior, whereas `sir` will use Sequential-Importance-
                Resampling using `init_strategy_num_candidates` to find init
                locations.
            device: torch device on which to compute, e.g. gpu, cpu.
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
            show_round_summary: Whether to show the validation loss and leakage after
                each round.
        """

        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def __call__(
        self,
        num_simulations: int,
        proposal: Optional[Any] = None,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        exclude_invalid_x: bool = True,
        discard_prior_samples: bool = False,
        retrain_from_scratch_each_round: bool = False,
    ) -> NeuralPosterior:
        r"""Run AALR / SNRE_A.

        Return posterior $p(\theta|x)$ after inference.

        Args:
            num_simulations: Number of simulator calls.
            proposal: Distribution that the parameters $\theta$ are drawn from.
                `proposal=None` uses the prior. Setting the proposal to a distribution
                targeted on a specific observation, e.g. a posterior $p(\theta|x_o)$
                obtained previously, can lead to less required simulations.
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. If None, we
                train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
                during training. Expect errors, silent or explicit, when `False`.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch_each_round: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.

        Returns:
            Posterior $p(\theta|x)$ that can be sampled and evaluated.
        """

        # AALR is defined for `num_atoms=2`.
        # Proxy to `super().__call__` to ensure right parameter.
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        return super().__call__(**kwargs, num_atoms=2)

    def _loss(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor:
        """
        Returns the binary cross-entropy loss for the trained classifier.

        The classifier takes as input a $(\theta,x)$ pair. It is trained to predict 1
        if the pair was sampled from the joint $p(\theta,x)$, and to predict 0 if the
        pair was sampled from the marginals $p(\theta)p(x)$.
        """

        assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = theta.shape[0]

        logits = self._classifier_logits(theta, x, num_atoms)
        likelihood = torch.sigmoid(logits).squeeze()

        # Alternating pairs where there is one sampled from the joint and one
        # sampled from the marginals. The first element is sampled from the
        # joint p(theta, x) and is labelled 1. The second element is sampled
        # from the marginals p(theta)p(x) and is labelled 0. And so on.
        labels = ones(2 * batch_size)  # two atoms
        labels[1::2] = 0.0

        # Binary cross entropy to learn the likelihood (AALR-specific)
        return nn.BCELoss()(likelihood, labels)
