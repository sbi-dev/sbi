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
    r"""Neural Likelihood Estimation (NLE) as in Papamakarios et al. (2019) [1].

    NLE trains a neural network to approximate the likelihood $p(x|\theta)$ using a
    conditional density estimator (normalizing flow). Unlike NPE methods, which directly
    estimate the posterior, NLE estimates the likelihood. Posterior sampling is then
    performed via MCMC (e.g., slice sampling) or rejection sampling.

    [1] Sequential Neural Likelihood: Fast Likelihood-free Inference with
        Autoregressive Flows, Papamakarios et al., AISTATS 2019,
        https://arxiv.org/abs/1805.07226

    Example:
    --------

    ::

        import torch
        from sbi.inference import NLE_A
        from sbi.utils import BoxUniform

        # 1. Setup prior and simulate data
        prior = BoxUniform(low=torch.zeros(3), high=torch.ones(3))
        theta = prior.sample((100,))
        x = theta + torch.randn_like(theta) * 0.1

        # 2. Train likelihood estimator
        inference = NLE_A(prior=prior)
        likelihood_estimator = inference.append_simulations(theta, x).train()

        # 3. Build posterior (uses MCMC for sampling)
        posterior = inference.build_posterior(likelihood_estimator)

        # 4. Sample from posterior
        x_o = torch.randn(1, 3)
        samples = posterior.sample((1000,), x=x_o)
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
