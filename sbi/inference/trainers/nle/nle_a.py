# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Literal, Optional, Union

from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.trainers.nle.nle_base import LikelihoodEstimatorTrainer
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimatorBuilder,
)
from sbi.sbi_types import Tracker
from sbi.utils.sbiutils import del_entries


class NLE_A(LikelihoodEstimatorTrainer):
    r"""Neural Likelihood Estimation (NLE-A) as in Papamakarios et al. (2019) [1].

    NLE-A trains a neural network to estimate the likelihood $p(x|\theta)$ using
    normalizing flows. Posterior sampling is performed via MCMC (e.g., slice
    sampling) or rejection sampling.

    [1] *Sequential Neural Likelihood: Fast Likelihood-free Inference with
        Autoregressive Flows*, Papamakarios et al., AISTATS 2019,
        https://arxiv.org/abs/1805.07226

    Example:
    --------

    .. code-block:: python

        import torch
        from sbi.inference import NLE_A
        from sbi.utils import BoxUniform

        prior = BoxUniform(low=torch.zeros(3), high=torch.ones(3))
        theta = prior.sample((100,))
        x = torch.randn(100, 10)

        inference = NLE_A(prior=prior)
        density_estimator = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(density_estimator)

        samples = posterior.sample((100,), x=x[0])
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[
            Literal["nsf", "maf", "mdn", "made"],
            ConditionalEstimatorBuilder[ConditionalDensityEstimator],
        ] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        tracker: Optional[Tracker] = None,
        show_progress_bars: bool = True,
    ):
        r"""Initialize Neural Likelihood Estimation.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. If `None`, the
                prior must be passed to `.build_posterior()`.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network, which adheres to
                `ConditionalEstimatorBuilder` protocol can be provided. The function
                will be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. The
                density estimator needs to provide the methods `.log_prob` and
                `.sample()` and must return a `ConditionalDensityEstimator`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: Deprecated alias for the TensorBoard summary writer.
                Use ``tracker`` instead.
            tracker: Tracking adapter used to log training metrics. If None, a
                TensorBoard tracker is used with a default log directory.
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
        """

        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)
