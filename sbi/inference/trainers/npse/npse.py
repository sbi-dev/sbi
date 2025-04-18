# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Literal, Optional, Union

from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.posteriors.vector_field_posterior import VectorFieldPosterior
from sbi.inference.trainers.npse.vector_field_inference import (
    VectorFieldEstimatorBuilder,
    VectorFieldInference,
)
from sbi.neural_nets.estimators import ConditionalVectorFieldEstimator
from sbi.neural_nets.factory import posterior_score_nn


class NPSE(VectorFieldInference):
    """Neural Posterior Score Estimation as in Geffner et al. and Sharrock et al."""

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        score_estimator: Union[str, VectorFieldEstimatorBuilder] = "mlp",
        sde_type: Literal["vp", "ve", "subvp"] = "ve",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        **kwargs,
    ):
        r"""
        Base class for Neural Posterior Score Estimation methods.

        Instead of performing conditonal *density* estimation, NPSE methods perform
        conditional *score* estimation i.e. they estimate the gradient of the log
        density using denoising score matching loss.

        NOTE: NPSE does not support multi-round inference with flexible proposals yet.
        You can try to run multi-round with truncated proposals, but note that this is
        not tested yet.

        Args:
            prior: Prior distribution.
            score_estimator: Neural network architecture for the
                vector field estimator. Can be a string (e.g. 'mlp' or 'ada_mlp') or a
                callable that implements the `VectorFieldEstimatorBuilder` protocol
                with `__call__` that receives `theta` and `x` and returns a
                `ConditionalVectorFieldEstimator`.
            sde_type: Type of SDE to use. Must be one of ['vp', 've', 'subvp'].
            device: Device to run the training on.
            logging_level: Logging level for the training. Can be an integer or a
                string.
            summary_writer: Tensorboard summary writer.
            show_progress_bars: Whether to show progress bars during training.
            kwargs: Additional keyword arguments passed to the default builder if
                `score_estimator` is a string.

        References:
            - Geffner, Tomas, George Papamakarios, and Andriy Mnih. "Score modeling for
                simulation-based inference." ICML 2023.
            - Sharrock, Louis, et al. "Sequential neural score estimation: Likelihood-
                free inference with conditional score based diffusion models." ICML 2024
        """
        super().__init__(
            prior=prior,
            vector_field_estimator_builder=score_estimator,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            sde_type=sde_type,
            **kwargs,
        )
        # score_estimator name is kept since it is public API, but it is
        # actually misleading since it is a builder for an estimator.

    def _build_default_nn_fn(self, **kwargs) -> VectorFieldEstimatorBuilder:
        net_type = kwargs.pop("vector_field_estimator_builder", "mlp")
        return posterior_score_nn(score_net_type=net_type, **kwargs)

    def build_posterior(
        self,
        vector_field_estimator: Optional[ConditionalVectorFieldEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "sde",
        **kwargs,
    ) -> VectorFieldPosterior:
        r"""Build posterior from the vector field estimator.

        Note that this is the same as the FMPE posterior, but the sample_with
        method is set to "sde" by default.

        For NPSE, the posterior distribution that is returned here implements
        the following functionality over the raw neural density estimator:
            - correct the calculation of the log probability such that
              samples outside of the prior bounds have log probability -inf.
            - reject samples that lie outside of the prior bounds.

        Args:
            vector_field_estimator: The vector field estimator that the posterior is
                based on. If `None`, use the latest vector field estimator that was
                trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Can be one of
                'sde' (default) or 'ode'. The 'sde' method uses the score to
                do a Langevin diffusion step, while the 'ode' method solves a
                probabilistic ODE with a numerical ODE solver.
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
