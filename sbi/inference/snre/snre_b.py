from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from sbi.inference.posterior import NeuralPosterior
from sbi.inference.snre.snre_base import RatioEstimator
from sbi.types import OneOrMore
from sbi.utils import del_entries


class SNRE_B(RatioEstimator):
    def __init__(
        self,
        simulator: Callable,
        prior,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        classifier: Union[str, Callable] = "resnet",
        mcmc_method: str = "slice_np",
        device: str = "cpu",
        logging_level: Union[int, str] = "warning",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        show_round_summary: bool = False,
    ):
        """SRE[1], here known as SNRE_B.

        [1] _On Contrastive Learning for Likelihood-free Inference_, Durkan et al.,
            ICML 2020, https://arxiv.org/pdf/2002.03712

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
            mcmc_method: Specify the method for MCMC sampling, either of:
                slice_np, slice, hmc, nuts.
            device: torch device on which to compute, e.g. gpu, cpu.
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A `SummaryWriter` to control, among others, log
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
        num_rounds: int,
        num_simulations_per_round: OneOrMore[int],
        x_o: Optional[Tensor] = None,
        num_atoms: int = 10,
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
        r"""Run SRE / SNRE_B.

        Return posterior $p(\theta|x)$ after inference (possibly over several rounds).

        Args:
            num_rounds: Number of rounds to run. Each round consists of a simulation and
                training phase. `num_rounds=1` leads to a posterior $p(\theta|x)$ valid
                for _any_ $x$ (amortized), but requires many simulations.
                Alternatively, with `num_rounds>1` the inference returns a posterior
                $p(\theta|x_o)$ focused on a specific observation `x_o`, potentially
                requiring less simulations.
            num_simulations_per_round: Number of simulator calls per round.
            x_o: An observation that is only required when doing inference
                over multiple rounds. After the first round, `x_o` is used to guide the
                sampling so that the simulator is run with parameters that are likely
                for that `x_o`, i.e. they are sampled from the posterior obtained in the
                previous round $p(\theta|x_o)$.
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
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        return super().__call__(**kwargs)

    def _loss(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor:
        """
        Return cross-entropy loss for 1-out-of-`num_atoms` classification.

        The classifier takes as input `num_atoms` $(\theta,x)$ pairs. Out of these
        pairs, one pair was sampled from the joint $p(\theta,x)$ and all others from the
        marginals $p(\theta)p(x)$. The classifier is trained to predict which of the
        pairs was sampled from the joint $p(\theta,x)$.
        """

        assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = theta.shape[0]
        logits = self._classifier_logits(theta, x, num_atoms)

        # For 1-out-of-`num_atoms` classification each datapoint consists
        # of `num_atoms` points, with one of them being the correct one.
        # We have a batch of `batch_size` such datapoints.
        logits = logits.reshape(batch_size, num_atoms)

        # Index 0 is the theta-x-pair sampled from the joint p(theta,x) and hence the
        # "correct" one for the 1-out-of-N classification.
        log_prob = logits[:, 0] - torch.logsumexp(logits, dim=-1)

        return -torch.mean(log_prob)
