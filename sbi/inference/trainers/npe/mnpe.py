# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Dict, Literal, Optional, Union

from torch.distributions import Distribution

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.posterior_parameters import (
    DirectPosteriorParameters,
    ImportanceSamplingPosteriorParameters,
    MCMCPosteriorParameters,
    RejectionPosteriorParameters,
    VIPosteriorParameters,
)
from sbi.inference.trainers.npe.npe_c import NPE_C
from sbi.neural_nets.estimators import MixedDensityEstimator
from sbi.neural_nets.estimators.base import ConditionalEstimatorBuilder
from sbi.sbi_types import TensorBoardSummaryWriter
from sbi.utils.sbiutils import del_entries


class MNPE(NPE_C):
    r"""Method that can infer discrete and continuous parameters (Mixed NPE).

    MNPE (Mixed Neural Posterior Estimation) is similar to NPE: it directly
    estimates a distribution over parameters given data. Unlike NPE, it is designed to
    be applied to parmaeters with mixed types, i.e., continuous and discrete parameters.
    This can occur, for example, in models with switching components. The emebedding
    net will only operate on the continuous parameters, note this to design the
    dimension of the embedding net.

    Example:
    --------

    ::

        import torch
        from sbi.inference import MNPE

        dim_theta_discrete = 3
        dim_theta_continuous = 2
        dim_theta = 5
        dim_x = 50

        num_sims = 100

        discrete_theta = torch.randint(low=0, high=2, size=(100, dim_theta_discrete))
        continuous_theta = torch.randn((num_sims, dim_theta_discrete))

        # Important: the theta must have all continuous paramters first, and
        # discrete parameters after this.
        theta = torch.cat([continuous_theta, discrete_theta], dim=1)
        x = torch.randn((num_sims, dim_x))

        inference = MNPE()
        _ = inference.append_simulations(theta, x).train()

        posterior = inference.build_posterior()

        x_o = torch.randn((1, dim_x))
        samples = posterior.sample((100,), x=x_o)
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[
            Literal["mnpe"],
            ConditionalEstimatorBuilder[MixedDensityEstimator],
        ] = "mnpe",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorBoardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""Initialize Mixed Neural Posterior Estimation (MNPE).

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. If `None`, the
                prior must be passed to `.build_posterior()`.
            density_estimator: If it is a string, it must be "mnpe" to use the
                preconfigured neural nets for MNPE. Alternatively, a function
                that builds a custom neural network, which adheres to
                `ConditionalEstimatorBuilder` protocol can be provided. The function
                will be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. The
                density estimator needs to provide the methods `.log_prob` and
                `.sample()` and must return a `MixedDensityEstimator`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
        """

        if isinstance(density_estimator, str):
            assert (
                density_estimator == "mnpe"
            ), f"""MNPE can be used with preconfigured 'mnpe' density estimator only,
                not with {density_estimator}."""
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def train(
        self,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        resume_training: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> MixedDensityEstimator:
        density_estimator = super().train(
            **del_entries(locals(), entries=("self", "__class__"))
        )
        assert isinstance(
            density_estimator, MixedDensityEstimator
        ), f"""Internal net must be of type
            MixedDensityEstimator but is {type(density_estimator)}."""
        return density_estimator

    def build_posterior(
        self,
        density_estimator: Optional[MixedDensityEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: Literal[
            "mcmc", "rejection", "vi", "importance", "direct"
        ] = "direct",
        mcmc_method: Literal[
            "slice_np",
            "slice_np_vectorized",
            "hmc_pyro",
            "nuts_pyro",
            "slice_pymc",
            "hmc_pymc",
            "nuts_pymc",
        ] = "slice_np_vectorized",
        vi_method: Literal["rKL", "fKL", "IW", "alpha"] = "rKL",
        direct_sampling_parameters: Optional[Dict[str, Any]] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        vi_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        importance_sampling_parameters: Optional[Dict[str, Any]] = None,
        posterior_parameters: Optional[
            Union[
                DirectPosteriorParameters,
                MCMCPosteriorParameters,
                VIPosteriorParameters,
                RejectionPosteriorParameters,
                ImportanceSamplingPosteriorParameters,
            ]
        ] = None,
    ) -> NeuralPosterior:
        """Build posterior from the neural density estimator.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`direct` | `mcmc` | `rejection` | `vi` | `importance`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`,
                `slice_np_vectorized`, `hmc_pyro`, `nuts_pyro`, `slice_pymc`,
                `hmc_pymc`, `nuts_pymc`. `slice_np` is a custom
                numpy implementation of slice sampling. `slice_np_vectorized` is
                identical to `slice_np`, but if `num_chains>1`, the chains are
                vectorized for `slice_np_vectorized` whereas they are run sequentially
                for `slice_np`. The samplers ending on `_pyro` are using Pyro, and
                likewise the samplers ending on `_pymc` are using PyMC.
            vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`].
            direct_sampling_parameters: Additional kwargs passed to `DirectPosterior`.
            mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
            vi_parameters: Additional kwargs passed to `VIPosterior`.
            rejection_sampling_parameters: Additional kwargs passed to `
                RejectionPosterior`.
            importance_sampling_parameters: Additional kwargs passed to
                `ImportanceSamplingPosterior`.
            posterior_parameters: Configuration passed to the init method for the
                posterior. Must be one of the following
                - `VIPosteriorParameters`
                - `ImportanceSamplingPosteriorParameters`
                - `MCMCPosteriorParameters`
                - `DirectPosteriorParameters`
                - `RejectionPosteriorParameters`

        Returns:
            Posterior $p(\theta|x)$ with `.sample()` and `.log_prob()` methods.
        """
        if density_estimator is not None:
            assert isinstance(
                density_estimator, MixedDensityEstimator
            ), f"""net must be of type MixedDensityEstimator but is {
                type(density_estimator)
            }."""

        return super().build_posterior(
            density_estimator=density_estimator,
            prior=prior,
            sample_with=sample_with,
            posterior_parameters=posterior_parameters,
            mcmc_method=mcmc_method,
            vi_method=vi_method,
            direct_sampling_parameters=direct_sampling_parameters,
            mcmc_parameters=mcmc_parameters,
            vi_parameters=vi_parameters,
            rejection_sampling_parameters=rejection_sampling_parameters,
            importance_sampling_parameters=importance_sampling_parameters,
        )
