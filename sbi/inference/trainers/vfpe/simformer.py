# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Dict, Literal, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.trainers.vfpe.base_vf_inference import (
    MaskedVectorFieldEstimatorBuilder,
    MaskedVectorFieldTrainer,
)
from sbi.neural_nets.estimators import MaskedConditionalVectorFieldEstimator
from sbi.neural_nets.factory import simformer_flow_nn, simformer_score_nn


class Simformer(MaskedVectorFieldTrainer):
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
    - if None, it will be equivalent to a full attention (i.e., full ones)
    mask, we suggest you to use None instead of ones to save memory resources

    NOTE:
        - Multi-round inference is not supported yet; the API is present for coherence
          with sbi.
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        mvf_estimator: Union[
            str,
            MaskedVectorFieldEstimatorBuilder,
        ] = "simformer",
        sde_type: Literal["vp", "ve", "subvp"] = "ve",
        posterior_latent_idx: Optional[list | Tensor] = None,
        posterior_observed_idx: Optional[list | Tensor] = None,
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
                this prior is ignored.
            mvf_estimator: Neural network architecture for the masked
                vector field estimator. Can be a string (e.g., `'simformer'`)
                or a callable that implements the `MaskedVectorFieldEstimatorBuilder`
                protocol. If a callable, `__call__` must accept `inputs`, and return
                a `MaskedConditionalVectorFieldEstimator`.
            sde_type: Type of SDE to use. Must be one of ['vp', 've', 'subvp'].
                NOTE: Only ve (variance exploding) is supported by now.
            posterior_latent_idx: List or Tensor of indexes identifying which
            variables are latent (to be infered),
            i.e, which variables identify $\theta$.
            posterior_observed_idx: List or Tensor of indexes identifying which
            variables are observed (to be infered) according to a posterior,
            i.e, which variables identify $x$.
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
            posterior_latent_idx=posterior_latent_idx,
            posterior_observed_idx=posterior_observed_idx,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            sde_type=sde_type,
            **kwargs,
        )

    def _build_default_nn_fn(self, **kwargs) -> MaskedVectorFieldEstimatorBuilder:
        net_type = kwargs.pop("vector_field_estimator_builder", "simformer")
        return simformer_score_nn(model=net_type, **kwargs)

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
        r"""Build conditinal distribution from the masked vector field estimator
        and the given fixed condition mask and edge mask.
        While the conditional generated can sample from an arbitary set of variables
        choosen by the user--defined through the condition mask, this method returns
        a NeuralPosterior object, despite the name such instance will work as expected
        and do not affect the capacities of the method.

        Args:
            condition_mask: A boolean mask indicating the role of each variable.
                Expected shape: `(batch_size, num_variables)`.
                - `True` (or `1`): The variable at this position is observed and its
                features will be used for conditioning.
                - `False` (or `0`): The variable at this position is latent and its
                features are subject to inference.
            edge_mask: A boolean mask defining the adjacency matrix of the directed
                acyclic graph (DAG) representing dependencies among variables.
                Expected shape: `(batch_size, num_variables, num_variables)`.
                - `True` (or `1`): An edge exists from the row variable to the column
                variable.
                - `False` (or `0`): No edge exists between these variables.
                - if None, it will be equivalent to a full attention (i.e., full ones)
                mask, we suggest you to use None instead of ones
                to save memory resources
            mvf_estimator: Neural network architecture for the masked
                vector field estimator. Can be a callable that implements
                the `MaskedVectorFieldEstimatorBuilder` protocol.
                If a callable, `__call__` must accept `inputs`, and return
                a `MaskedConditionalVectorFieldEstimator`. If None uses the
                underlying trained neural net.
            prior: Prior distribution. Its primary use is for rejecting samples that
                fall outside its defined support. For the core inference process,
                this prior is ignored, as the actual "prior" over which the diffusion
                model operates is standard Gaussian noise.
            sample_with: Method to use for sampling from the posterior. Can be one of
                'sde' (default) or 'ode'. The 'sde' method uses the score to
                do a Langevin diffusion step, while the 'ode' method solves a
                probabilistic ODE with a numerical ODE solver.
            vectorfield_sampling_parameters: Additional keyword arguments passed to
                `VectorFieldPosterior`.
            **kwargs: Additional keyword arguments passed to
                `VectorFieldBasedPotential`.
        Returns:
            Posterior $p(\text{latent}|\text{observed})$
                with `.sample()` and `.log_prob()` methods.
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
        r"""Build posterior from the masked vector field estimator given
        the latent and observed indexes passed at init time (or updated
        later)

        Args:
            edge_mask: A boolean mask defining the adjacency matrix of the directed
                acyclic graph (DAG) representing dependencies among variables.
                Expected shape: `(batch_size, num_variables, num_variables)`.
                - `True` (or `1`): An edge exists from the row variable to the column
                variable.
                - `False` (or `0`): No edge exists between these variables.
                - if None, it will be equivalent to a full attention (i.e., full ones)
                mask, we suggest you to use None instead of ones
                to save memory resources
            mvf_estimator: Neural network architecture for the masked
                vector field estimator. Can be a callable that implements
                the `MaskedVectorFieldEstimatorBuilder` protocol.
                If a callable, `__call__` must accept `inputs`, and return
                a `MaskedConditionalVectorFieldEstimator`. If None uses the
                underlying trained neural net.
            prior: Prior distribution. Its primary use is for rejecting samples that
                fall outside its defined support. For the core inference process,
                this prior is ignored, as the actual "prior" over which the diffusion
                model operates is standard Gaussian noise.
            sample_with: Method to use for sampling from the posterior. Can be one of
                'sde' (default) or 'ode'. The 'sde' method uses the score to
                do a Langevin diffusion step, while the 'ode' method solves a
                probabilistic ODE with a numerical ODE solver.
            vectorfield_sampling_parameters: Additional keyword arguments passed to
                `VectorFieldPosterior`.
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
        r"""Build likelihood from the masked vector field estimator given
        the latent and observed indexes passed at init time (or updated
        later).

        This method simply consists in calling build_posterior on the negative
        of the condition masked generated from the latent and observed indexes.

        Args:
            edge_mask: A boolean mask defining the adjacency matrix of the directed
                acyclic graph (DAG) representing dependencies among variables.
                Expected shape: `(batch_size, num_variables, num_variables)`.
                - `True` (or `1`): An edge exists from the row variable to the column
                variable.
                - `False` (or `0`): No edge exists between these variables.
                - if None, it will be equivalent to a full attention (i.e., full ones)
                mask, we suggest you to use None instead of ones
                to save memory resources
             mvf_estimator: Neural network architecture for the masked
                vector field estimator. Can be a callable that implements
                the `MaskedVectorFieldEstimatorBuilder` protocol.
                If a callable, `__call__` must accept `inputs`, and return
                a `MaskedConditionalVectorFieldEstimator`. If None uses the
                underlying trained neural net.
            prior: Prior distribution. Its primary use is for rejecting samples that
                fall outside its defined support. For the core inference process,
                this prior is ignored, as the actual "prior" over which the diffusion
                model operates is standard Gaussian noise.
            sample_with: Method to use for sampling from the posterior. Can be one of
                'sde' (default) or 'ode'. The 'sde' method uses the score to
                do a Langevin diffusion step, while the 'ode' method solves a
                probabilistic ODE with a numerical ODE solver.
            vectorfield_sampling_parameters: Additional keyword arguments passed to
                `VectorFieldPosterior`.
            **kwargs: Additional keyword arguments passed to
                `VectorFieldBasedPotential`.
        Returns:
            Likelihood $p(x|\theta)$  with `.sample()` and `.log_prob()` methods.
        """

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


class FlowMatchingSimformer(Simformer):
    """Flow-matching version of the Simformer as in Gloeckler et al. (2024).

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

    NOTE:
        - Multi-round inference is not supported yet; the API is present for coherence
            with sbi.
    """

    def _build_default_nn_fn(self, **kwargs) -> MaskedVectorFieldEstimatorBuilder:
        model = kwargs.pop("vector_field_estimator_builder", "simformer")
        kwargs.pop("sde_type", None)  # sde_type is not used in FM Simformer
        return simformer_flow_nn(model=model, **kwargs)
