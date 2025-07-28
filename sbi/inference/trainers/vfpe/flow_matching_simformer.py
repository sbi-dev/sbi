# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License v2.0, see <https://www.apache.org/licenses/LICENSE-2.0>.


from typing import Any, Dict, Literal, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter

from sbi import utils as utils
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.trainers.vfpe.base_vf_inference import (
    MaskedVectorFieldEstimatorBuilder,
    MaskedVectorFieldTrainer,
)
from sbi.neural_nets.estimators.base import (
    MaskedConditionalVectorFieldEstimator,
)
from sbi.neural_nets.factory import simformer_flow_nn


class FlowMatchingSimformer(MaskedVectorFieldTrainer):
    """Flow Matching Posterior Estimation (FMPE)."""

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        mvf_estimator: Union[
            str,
            MaskedVectorFieldEstimatorBuilder,
        ] = "simformer",
        latent_idx: Optional[list | Tensor] = None,
        observed_idx: Optional[list | Tensor] = None,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        **kwargs,
    ) -> None:
        """Initialization method for the FMPE class.

        Args:
            prior: Prior distribution.
            density_estimator: Deprecated. Use `vf_estimator` instead.
            vf_estimator: Neural network architecture used to learn the
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
            mvf_estimator_builder=mvf_estimator,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            **kwargs,
        )

        self.latent_idx = (
            torch.as_tensor(latent_idx, dtype=torch.long)
            if latent_idx is not None
            else None
        )
        self.observed_idx = (
            torch.as_tensor(observed_idx, dtype=torch.long)
            if observed_idx is not None
            else None
        )

    def _build_default_nn_fn(self, **kwargs) -> MaskedVectorFieldEstimatorBuilder:
        model = kwargs.pop("vector_field_estimator_builder", "simformer")
        return simformer_flow_nn(model=model, **kwargs)

    def build_conditional(
        self,
        condition_mask: Union[Tensor, list],
        edge_mask: Optional[Tensor] = None,
        mvf_estimator: Optional[MaskedConditionalVectorFieldEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: Literal['ode', 'sde'] = "sde",
        vectorfield_sampling_parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> NeuralPosterior:
        r"""Build posterior from the masked vector field estimator and given
        fixed condition mask and edge mask.

        Args:
            condition_mask: A boolean mask indicating the role of each node.
                If no condition mask is provided, the latent and observed indexes
                passed at init time will be used. If no such indexes were passed before
                an error will raise.
                Expected shape: `(batch_size, num_nodes)`.
                - `True` (or `1`): The node at this position is observed and its
                    features will be used for conditioning.
                - `False` (or `0`): The node at this position is latent and its
                    parameters are subject to inference.
            edge_mask: A boolean mask defining the adjacency matrix of the directed
                acyclic graph (DAG) representing dependencies among nodes.
                Expected shape: `(batch_size, num_nodes, num_nodes)`.
                - `True` (or `1`): An edge exists from the row node to the column node.
                - `False` (or `0`): No edge exists between these nodes.
            prior: Prior distribution. Its primary use is for rejecting samples that
                fall outside its defined support. For the core inference process,
                this prior is ignored, as the actual "prior" over which the diffusion
                model operates is standard Gaussian noise.
            sample_with: Method to use for sampling from the posterior. Can be one of
                'sde' (default) or 'ode'. The 'sde' method uses the score to
                do a Langevin diffusion step, while the 'ode' method solves a
                probabilistic ODE with a numerical ODE solver.
            **kwargs: Additional keyword arguments passed to
                `VectorFieldBasedPotential`.
        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods.
        """

        # If the condtion mask is a list we must convert it
        condition_mask = torch.as_tensor(condition_mask)
        return self._build_conditional(
            condition_mask=condition_mask,
            edge_mask=edge_mask,
            mvf_estimator=mvf_estimator,
            prior=prior,
            sample_with=sample_with,
            vectorfield_sampling_parameters=vectorfield_sampling_parameters,
            **kwargs,
        )

    def build_posterior(
        self,
        edge_mask: Optional[Tensor] = None,
        mvf_estimator: Optional[MaskedConditionalVectorFieldEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: Literal['ode', 'sde'] = "sde",
        vectorfield_sampling_parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> NeuralPosterior:
        r"""Build posterior from the masked vector field estimator and given
        fixed condition mask and edge mask.

        Args:
            condition_masks: A boolean mask indicating the role of each node.
                If no condition mask is provided, the latent and observed indexes
                passed at init time will be used. If no such indexes were passed before
                an error will raise.
                Expected shape: `(batch_size, num_nodes)`.
                - `True` (or `1`): The node at this position is observed and its
                    features will be used for conditioning.
                - `False` (or `0`): The node at this position is latent and its
                    parameters are subject to inference.
            edge_masks: A boolean mask defining the adjacency matrix of the directed
                acyclic graph (DAG) representing dependencies among nodes.
                Expected shape: `(batch_size, num_nodes, num_nodes)`.
                - `True` (or `1`): An edge exists from the row node to the column node.
                - `False` (or `0`): No edge exists between these nodes.
            prior: Prior distribution. Its primary use is for rejecting samples that
                fall outside its defined support. For the core inference process,
                this prior is ignored, as the actual "prior" over which the diffusion
                model operates is standard Gaussian noise.
            sample_with: Method to use for sampling from the posterior. Can be one of
                'sde' (default) or 'ode'. The 'sde' method uses the score to
                do a Langevin diffusion step, while the 'ode' method solves a
                probabilistic ODE with a numerical ODE solver.
            **kwargs: Additional keyword arguments passed to
                `VectorFieldBasedPotential`.
        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods.
        """

        # Indexes for condition were provided at init
        condition_mask = self._generate_posterior_condition_mask()

        return self.build_conditional(
            condition_mask=condition_mask,
            edge_mask=edge_mask,
            mvf_estimator=mvf_estimator,
            prior=prior,
            sample_with=sample_with,
            vectorfield_sampling_parameters=vectorfield_sampling_parameters,
            **kwargs,
        )

    def build_likelihood(
        self,
        edge_mask: Optional[Tensor] = None,
        mvf_estimator: Optional[MaskedConditionalVectorFieldEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: Literal['ode', 'sde'] = "sde",
        vectorfield_sampling_parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> NeuralPosterior:
        # Indexes for condition were provided at init
        condition_mask = ~self._generate_posterior_condition_mask()

        return self.build_conditional(
            condition_mask=condition_mask,
            edge_mask=edge_mask,
            mvf_estimator=mvf_estimator,
            prior=prior,
            sample_with=sample_with,
            vectorfield_sampling_parameters=vectorfield_sampling_parameters,
            **kwargs,
        )

    def set_condition_indexes(
        self, new_latent_idx: Union[list, Tensor], new_observed_idx: Union[list, Tensor]
    ):
        self.latent_idx = torch.as_tensor(new_latent_idx, dtype=torch.long)
        self.observed_idx = torch.as_tensor(new_observed_idx, dtype=torch.long)

    def _generate_posterior_condition_mask(self):
        if self.latent_idx is None or self.observed_idx is None:
            raise ValueError(
                "You did not pass latent and observed variable indexes. "
                "sbi cannot generate a posterior or likelihood without any knowledge"
                "of which variables are latent or observed "
                "If you already instanciated a Masked Vector Filed Inference "
                "and would like to update the current conditon indexes, "
                "you can use the setter function `set_condtion_indexes()`"
            )
        return self.generate_condition_mask_from_idx(self.latent_idx, self.observed_idx)

    @staticmethod
    def generate_condition_mask_from_idx(
        latent_idx: Union[list, Tensor],
        observed_idx: Union[list, Tensor],
    ) -> Tensor:
        latent_idx = torch.as_tensor(latent_idx, dtype=torch.long)
        observed_idx = torch.as_tensor(observed_idx, dtype=torch.long)

        # Check for overlap
        if torch.any(torch.isin(latent_idx, observed_idx)):
            raise ValueError(
                f"latent_idx and observed_idx must be disjoint, "
                f"but you provided {latent_idx=} and {observed_idx=}."
            )

        all_idx = torch.cat([latent_idx, observed_idx])
        unique_idx = torch.unique(all_idx)
        num_nodes = unique_idx.numel()
        # Check for completeness
        if not torch.equal(torch.sort(unique_idx).values, torch.arange(num_nodes)):
            raise ValueError(
                f"latent_idx and observed_idx together must cover a complete range of "
                f"integers from 0 to N-1 without gaps."
                f"but you provided {latent_idx=} and {observed_idx=}."
            )

        # If checks pass we can generate the condition mask
        condition_mask = torch.zeros(num_nodes, dtype=torch.bool)
        condition_mask[latent_idx] = False
        condition_mask[observed_idx] = True
        return condition_mask
