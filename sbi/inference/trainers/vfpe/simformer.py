# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Literal, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.posteriors.vector_field_posterior import VectorFieldPosterior
from sbi.inference.trainers.vfpe.base_vf_inference import (
    MaskedVectorFieldEstimatorBuilder,
    MaskedVectorFieldInference,
)
from sbi.neural_nets.estimators import MaskedConditionalVectorFieldEstimator
from sbi.neural_nets.factory import simformer_nn


class Simformer(MaskedVectorFieldInference):
    """Simformer as in Gloeckler et al. (2024).

    Simformer enables sampling from arbitrary conditional joint distributions,
    not just posterior or likelihood, by operating on a unified input tensor
    that represents all variables.

    The roles of variables—latent (to be inferred) or observed (to be conditioned on)—
    are specified by a boolean mask `condition_mask`.
    - `True` (or `1`): The variable is observed (conditioned on).
    - `False` (or `0`): The variable is latent (to be inferred).

    Dependencies among variables are defined by a boolean adjacency matrix `edge_mask`.
    - `True` (or `1`): An edge exists from the row variable to the column variable.
    - `False` (or `0`): No edge exists.

    For posterior inference $p(\\theta|x)$, set theta variables as latent and
    data (x) variables as observed in `condition_mask`.

    NOTE:
        - Multi-round inference is not supported yet; the API is present for coherence
          with sbi.
        - The `prior` argument is currently only used for sample rejection in cases
          where the inferred variables fall outside expected support.
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        mvf_estimator: Union[
            str,
            MaskedVectorFieldEstimatorBuilder,
        ] = "simformer",
        sde_type: Literal["vp", "ve", "subvp"] = "ve",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        **kwargs,
    ):
        r"""Initialize Simformer.

        Args:
            prior: Prior distribution. Its primary use is for rejecting samples that
                fall outside its defined support. For the core inference process,
                this prior is ignored, as the actual "prior" over which the diffusion
                model operates is standard Gaussian noise.
            mvf_estimator: Neural network architecture for the masked
                vector field estimator. Can be a string (e.g., `'simformer'`)
                or a callable that implements the `MaskedVectorFieldEstimatorBuilder`
                protocol. If a callable, `__call__` must accept `inputs`, and return
                a `MaskedConditionalVectorFieldEstimator`.
            sde_type: Type of SDE to use. Must be one of ['vp', 've', 'subvp'].
                NOTE: Only ve (variance exploding) is supported by now.
            device: Device to run the training on.
            logging_level: Logging level for the training. Can be an integer or a
                string.
            summary_writer: Tensorboard summary writer.
            show_progress_bars: Whether to show progress bars during training.
            kwargs: Additional keyword arguments passed to the default builder if
                `score_estimator` is a string.

        References:
            - Gloeckler, Deistler, Weilbach, Wood, Macke.
                "All-in-one simulation-based inference.", ICML 2024
        """
        super().__init__(
            prior=prior,
            mvf_estimator_builder=mvf_estimator,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            sde_type=sde_type,
            **kwargs,
        )

    def _build_default_nn_fn(self, **kwargs) -> MaskedVectorFieldEstimatorBuilder:
        net_type = kwargs.pop("vector_field_estimator_builder", "simformer")
        return simformer_nn(model=net_type, **kwargs)

    def build_posterior(
        self,
        condition_mask: Tensor,
        edge_mask: Optional[Tensor] = None,
        mvf_estimator: Optional[MaskedConditionalVectorFieldEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "sde",
        **kwargs,
    ) -> VectorFieldPosterior:
        r"""Build posterior from the masked vector field estimator and given
        fixed condition mask and edge mask.

        Args:
            condition_masks: A boolean mask indicating the role of each node.
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

        batch_dims = condition_mask.shape[:-1]
        num_nodes = condition_mask.shape[-1]
        if edge_mask is None:
            edge_mask = torch.ones((num_nodes, num_nodes)).bool()
            edge_mask = edge_mask.repeat(*batch_dims, 1, 1)

        return self._build_posterior(
            condition_mask,
            edge_mask,
            mvf_estimator=mvf_estimator,
            prior=prior,
            sample_with=sample_with,
            **kwargs,
        )

    def build_likelihood(
        self,
        conditional_mask: Tensor,
        edge_mask: Optional[Tensor] = None,
        mvf_estimator: Optional[MaskedConditionalVectorFieldEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "sde",
    ):
        raise NotImplementedError
