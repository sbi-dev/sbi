# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License v2.0, see <https://www.apache.org/licenses/LICENSE-2.0>.


from typing import Optional, Union

from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter

from sbi import utils as utils
from sbi.inference.posteriors.vector_field_posterior import VectorFieldPosterior
from sbi.inference.trainers.npse.vector_field_inference import (
    VectorFieldEstimatorBuilder,
    VectorFieldInference,
)
from sbi.neural_nets import flowmatching_nn
from sbi.neural_nets.estimators import ConditionalVectorFieldEstimator


class FMPE(VectorFieldInference):
    """
    Implements the Flow Matching Posterior Estimator (FMPE) for sbi.
    """

    def __init__(
        self,
        prior: Optional[Distribution],
        density_estimator: Union[str, VectorFieldEstimatorBuilder] = "mlp",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        **kwargs,
    ) -> None:
        """Initialization method for the FMPE class.

        Args:
            prior: Prior distribution.
            density_estimator: Neural network architecture used to learn the
                vector field estimator. Can be a string (e.g. 'mlp' or 'ada_mlp') or a
                callable that implements the `VectorFieldEstimatorBuilder` protocol
                with `__call__` that receives `theta` and `x` and returns a
                `ConditionalVectorFieldEstimator`.
            device: Device to use for training.
            logging_level: Logging level.
            summary_writer: Summary writer for tensorboard.
            show_progress_bars: Whether to show progress bars.
            **kwargs: Additional keyword arguments passed to the default builder if
                `density_estimator` is a string.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            vector_field_estimator_builder=density_estimator,
            **kwargs,
        )
        # density_estimator name is kept since it is public API, but it is
        # actually misleading since it is a builder for an estimator.

    def _build_default_nn_fn(self, **kwargs) -> VectorFieldEstimatorBuilder:
        model = kwargs.pop("vector_field_estimator_builder", "mlp")
        return flowmatching_nn(model=model, **kwargs)

    def build_posterior(
        self,
        vector_field_estimator: Optional[ConditionalVectorFieldEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "ode",
        **kwargs,
    ) -> VectorFieldPosterior:
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
            **kwargs: Additional keyword arguments passed to
                `VectorFieldBasedPotential`.


        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods.
        """
        return self._build_posterior(
            vector_field_estimator=vector_field_estimator,
            prior=prior,
            sample_with=sample_with,
            **kwargs,
        )
