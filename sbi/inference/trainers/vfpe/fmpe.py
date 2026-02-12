# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import warnings
from typing import Any, Dict, Literal, Optional, Union

from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter

from sbi import utils as utils
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.posterior_parameters import VectorFieldPosteriorParameters
from sbi.inference.trainers.vfpe.base_vf_inference import (
    VectorFieldTrainer,
)
from sbi.neural_nets.estimators.base import (
    ConditionalEstimatorBuilder,
    ConditionalVectorFieldEstimator,
)
from sbi.neural_nets.factory import posterior_flow_nn
from sbi.sbi_types import Tracker


class FMPE(VectorFieldTrainer):
    r"""Flow Matching Posterior Estimation (FMPE) [1].

    FMPE trains a continuous normalizing flow (CNF) to transform samples from the
    prior distribution to the posterior distribution using flow matching. Instead of
    maximum likelihood, it trains a vector field to match the marginal vector field
    of a conditional flow that interpolates between the prior and posterior. Sampling
    is performed by solving an ODE.

    [1] Flow Matching for Generative Modeling, Lipman et al., ICLR 2023,
        https://arxiv.org/abs/2210.02747

    Example:
    --------

    ::

        import torch
        from sbi.inference import FMPE
        from sbi.utils import BoxUniform

        # 1. Setup prior and simulate data
        prior = BoxUniform(low=torch.zeros(3), high=torch.ones(3))
        theta = prior.sample((100,))
        x = theta + torch.randn_like(theta) * 0.1

        # 2. Train flow matching estimator
        inference = FMPE(prior=prior)
        flow_estimator = inference.append_simulations(theta, x).train()

        # 3. Build posterior (uses ODE solver for sampling)
        posterior = inference.build_posterior(flow_estimator)

        # 4. Sample from posterior
        x_o = torch.randn(1, 3)
        samples = posterior.sample((1000,), x=x_o)
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        vf_estimator: Union[
            Literal["mlp", "ada_mlp", "transformer", "transformer_cross_attn"],
            ConditionalEstimatorBuilder[ConditionalVectorFieldEstimator],
        ] = "mlp",
        density_estimator: Optional[
            ConditionalEstimatorBuilder[ConditionalVectorFieldEstimator]
        ] = None,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        tracker: Optional[Tracker] = None,
        show_progress_bars: bool = True,
        **kwargs,
    ) -> None:
        """Initialization method for the FMPE class.

        Args:
            prior: Prior distribution.
            vf_estimator: Neural network architecture used to learn the
                vector field estimator. Can be a string (e.g. 'mlp', 'ada_mlp',
                'transformer' or 'transformer_cross_attn') or a callable that implements
                the `ConditionalEstimatorBuilder` protocol with `__call__` that receives
                `theta` and `x` and returns a `ConditionalVectorFieldEstimator`.
            density_estimator: Deprecated. Use `vf_estimator` instead. When passed, a
                warning is raised and the `vf_estimator="mlp"` default is used.
            device: Device to use for training.
            logging_level: Logging level.
            summary_writer: Deprecated alias for the TensorBoard summary writer.
                Use ``tracker`` instead.
            tracker: Tracking adapter used to log training metrics. If None, a
                TensorBoard tracker is used with a default log directory.
            show_progress_bars: Whether to show progress bars.
            **kwargs: Additional keyword arguments passed to the default builder if
                `density_estimator` is a string.
        """

        if density_estimator is not None:
            warnings.warn(
                "`density_estimator` is deprecated and will be removed in a future "
                "release. Use `vf_estimator` instead.",
                FutureWarning,
                stacklevel=2,
            )
            vf_estimator = density_estimator

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            tracker=tracker,
            show_progress_bars=show_progress_bars,
            vector_field_estimator_builder=vf_estimator,
            **kwargs,
        )

    def build_posterior(
        self,
        vector_field_estimator: Optional[ConditionalVectorFieldEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: Literal["ode", "sde"] = "ode",
        vectorfield_sampling_parameters: Optional[Dict[str, Any]] = None,
        posterior_parameters: Optional[VectorFieldPosteriorParameters] = None,
    ) -> NeuralPosterior:
        r"""Build posterior from the flow matching estimator.

        Note that this is the same as the NPSE posterior, but the sample_with method
        is set to "ode" by default.

        For FMPE, the posterior distribution that is returned here implements
        the following functionality over the raw neural density estimator:
            - correct the calculation of the log probability such that
              samples outside of the prior bounds have log probability -inf.
            - reject samples that lie outside of the prior bounds.

        Args:
            vector_field_estimator: The flow matching estimator that
                the posterior is based on. If `None`, use the latest neural
                flow matching estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior.
                Can be one of 'sde' (default) or 'ode'. The 'sde' method uses
                the score to do a Langevin diffusion step, while the 'ode' method
                uses the score to define a probabilistic ODE and solves it with
                a numerical ODE solver.
            vectorfield_sampling_parameters: Additional keyword arguments passed to
                `VectorFieldPosterior`.
            posterior_parameters: Configuration passed to the init method for
                VectorFieldPosterior.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods.
        """

        return super().build_posterior(
            estimator=vector_field_estimator,
            prior=prior,
            sample_with=sample_with,
            vectorfield_sampling_parameters=vectorfield_sampling_parameters,
            posterior_parameters=posterior_parameters,
        )

    def _build_default_nn_fn(
        self,
        model: Literal["mlp", "ada_mlp", "transformer", "transformer_cross_attn"],
        **kwargs,
    ) -> ConditionalEstimatorBuilder[ConditionalVectorFieldEstimator]:
        return posterior_flow_nn(model=model, **kwargs)
