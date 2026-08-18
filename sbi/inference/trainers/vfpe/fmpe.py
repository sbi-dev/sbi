# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import warnings
from typing import Any, ClassVar, Dict, Literal, Optional, Union

from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter

from sbi import utils as utils
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.posterior_parameters import VectorFieldPosteriorParameters
from sbi.inference.trainers.vfpe.base_vf_inference import (
    VectorFieldTrainer,
)
from sbi.neural_nets.estimators.base import (
    ConditionalEstimatorBuildFn,
    ConditionalVectorFieldEstimator,
)
from sbi.neural_nets.factory import posterior_flow_nn
from sbi.neural_nets.net_builders.estimator_configs import (
    VF_MODELS,
)
from sbi.neural_nets.net_builders.vector_field_nets import FlowMatchingConfig
from sbi.sbi_types import Tracker


class FMPE(VectorFieldTrainer):
    r"""Flow Matching Posterior Estimation (FMPE) [1].

    FMPE trains a continuous normalizing flow (CNF) to transform samples from the
    prior distribution to the posterior distribution using flow matching. Instead of
    maximum likelihood, it trains a vector field to match the marginal vector field
    of a conditional flow that interpolates between the prior and posterior. The
    neural network architecture for the vector field is not constrained like for
    flows and can be any expressive network. Sampling is performed by solving an ODE,
    which can be slower than flow-based NPE, but log_prob evaluation can also be slower.

    NOTE: FMPE does not support multi-round inference with flexible proposals yet.
    You can try multi-round with truncated proposals, but this is not tested.

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

    _ALLOWED_CONFIG_TYPE: ClassVar[type] = FlowMatchingConfig

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        vf_estimator: Union[
            VF_MODELS,
            FlowMatchingConfig,
            ConditionalEstimatorBuildFn[ConditionalVectorFieldEstimator],
            None,
        ] = None,
        density_estimator: Optional[
            ConditionalEstimatorBuildFn[ConditionalVectorFieldEstimator]
        ] = None,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        tracker: Optional[Tracker] = None,
        show_progress_bars: bool = True,
    ) -> None:
        """Initialization method for the FMPE class.

        Args:
            prior: Prior distribution.
            vf_estimator: The vector-field estimator used for flow-matching
                inference. If ``None`` (default), uses a
                ``FlowMatchingConfig`` with default settings. A
                FlowMatchingConfig can be passed to configure
                the estimator. If it is a string (deprecated), use a
                pre-configured network of the provided type (one of
                mlp, ada_mlp, transformer, transformer_cross_attn).
                Alternatively, a function that builds a custom neural
                network can be provided.
            density_estimator: Deprecated. Use `vf_estimator` instead.
            device: Device to use for training.
            logging_level: Logging level.
            summary_writer: Deprecated alias for the TensorBoard summary writer.
                Use ``tracker`` instead.
            tracker: Tracking adapter used to log training metrics. If None, a
                TensorBoard tracker is used with a default log directory.
            show_progress_bars: Whether to show progress bars.
        """

        if density_estimator is not None:
            if vf_estimator is not None:
                raise ValueError(
                    "Cannot pass both `density_estimator` and `vf_estimator`. "
                    "Use `vf_estimator` only; `density_estimator` is deprecated."
                )
            warnings.warn(
                "`density_estimator` is deprecated and will be removed in a future "
                "release. Use `vf_estimator` instead.",
                FutureWarning,
                stacklevel=2,
            )
            vf_estimator = density_estimator

        if vf_estimator is None:
            vf_estimator = FlowMatchingConfig()
        elif isinstance(vf_estimator, str):
            warnings.warn(
                "Passing a string for `vf_estimator` is deprecated. "
                "Use FlowMatchingConfig() instead, e.g. "
                "`from sbi.neural_nets import FlowMatchingConfig`.",
                FutureWarning,
                stacklevel=2,
            )

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            tracker=tracker,
            show_progress_bars=show_progress_bars,
            vector_field_estimator_builder=vf_estimator,
        )

        # When vf_estimator is a string (deprecated path), build the default.
        if isinstance(vf_estimator, str):
            self._build_neural_net = self._build_default_nn_fn(model=vf_estimator)

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
                Can be one of 'ode' (default) or 'sde'. The 'ode' method uses
                the velocity field to define a probabilistic ODE and solves it
                with a numerical ODE solver. The 'sde' method uses the score to
                do a Langevin diffusion step.
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
        model: VF_MODELS,
    ) -> ConditionalEstimatorBuildFn[ConditionalVectorFieldEstimator]:
        return posterior_flow_nn(model=model)
