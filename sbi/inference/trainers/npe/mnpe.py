# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Callable, Dict, Optional, Union

from torch.distributions import Distribution

from sbi.inference.posteriors import (
    DirectPosterior,
    ImportanceSamplingPosterior,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
)
from sbi.inference.trainers.npe.npe_c import NPE_C
from sbi.neural_nets.estimators import MixedDensityEstimator
from sbi.sbi_types import TensorboardSummaryWriter
from sbi.utils.sbiutils import del_entries


class MNPE(NPE_C):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "mnpe",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""Mixed Neural Posterior Estimation (MNPE).

        Like NPE-C, but designed to be applied to data with mixed types, e.g.,
        continuous parameters and discrete parameters like they occur in models with
        switching components. The emebedding net will only operate on the continuous
        parameters, note this to design the dimension of the embedding net.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. If `None`, the
                prior must be passed to `.build_posterior()`.
            density_estimator: If it is a string, it must be "mnpe" to use the
                preconfigured neural nets for MNPE. Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob`, `.log_prob_iid()` and `.sample()`.
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
        sample_with: str = "direct",
        mcmc_method: str = "slice_np_vectorized",
        vi_method: str = "rKL",
        direct_sampling_parameters: Optional[Dict[str, Any]] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        vi_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        importance_sampling_parameters: Optional[Dict[str, Any]] = None,
    ) -> Union[
        MCMCPosterior,
        RejectionPosterior,
        VIPosterior,
        DirectPosterior,
        ImportanceSamplingPosterior,
    ]:
        """Build posterior from the neural density estimator.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`direct` | `mcmc` | `rejection` | `vi` | `importance`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`].
            direct_sampling_parameters: Additional kwargs passed to `DirectPosterior`.
            mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
            vi_parameters: Additional kwargs passed to `VIPosterior`.
            rejection_sampling_parameters: Additional kwargs passed to `
                RejectionPosterior`.
            importance_sampling_parameters: Additional kwargs passed to
                `ImportanceSamplingPosterior`.

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
            mcmc_method=mcmc_method,
            vi_method=vi_method,
            direct_sampling_parameters=direct_sampling_parameters,
            mcmc_parameters=mcmc_parameters,
            vi_parameters=vi_parameters,
            rejection_sampling_parameters=rejection_sampling_parameters,
            importance_sampling_parameters=importance_sampling_parameters,
        )
