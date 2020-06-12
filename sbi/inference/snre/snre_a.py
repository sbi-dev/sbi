from __future__ import annotations

from typing import Optional, Callable, Union
from sbi.types import OneOrMore

import torch
from torch import nn, ones, Tensor
from torch.utils.tensorboard import SummaryWriter

from sbi.inference.posterior import NeuralPosterior
from sbi.inference.snre.snre_base import RatioEstimator
from sbi.utils import del_entries
from sbi.utils.torchutils import get_default_device


class SNRE_A(RatioEstimator):
    def __init__(
        self,
        simulator: Callable,
        prior,
        x_shape: Optional[torch.Size] = None,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        embedding_net: nn.Module = nn.Identity(),
        classifier: Union[str, nn.Module] = "resnet",
        mcmc_method: str = "slice_np",
        device: Union[torch.device, str] = get_default_device(),
        logging_level: Union[int, str] = "warning",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        show_round_summary: bool = False,
    ):
        """AALR[1], here known as SNRE_A.

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
            x_shape: Shape of a single simulation output $x$, has to be (1,N).
            num_workers: Number of parallel workers to use for simulations.
            simulation_batch_size: Number of parameter sets that the simulator
                maps to data x at once. If None, we simulate all parameter sets at the
                same time. If >= 1, the simulator has to process data of shape
                (simulation_batch_size, parameter_dimension).
            embedding_net: Can be used to encode observations $x$. Currently not
                implemented.
            classifier: Classifier trained to approximate likelihood rations. If str,
                use a pre-configured neural network.
            mcmc_method: If MCMC sampling is used, specify the method here: either of
                slice_np, slice, hmc, nuts.
            device: torch device on which to compute, e.g. cuda, cpu.
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
        batch_size: int = 50,
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
            batch_size: Training batch size.
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
