# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Literal, Optional, Union

from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.trainers.vfpe.base_vf_inference import MaskedVectorFieldTrainer
from sbi.neural_nets.estimators.base import (
    MaskedConditionalEstimatorBuilder,
    MaskedConditionalVectorFieldEstimator,
)
from sbi.neural_nets.factory import simformer_flow_nn, simformer_score_nn


class Simformer(MaskedVectorFieldTrainer):
    """Simformer as in Gloeckler et al. (ICML, 2024) [1].

    Simformer enables sampling from arbitrary conditional distributions, not just
    posterior or likelihood, by operating on a unified input tensor that represents all
    variables.

    The roles of variables—latent (to be inferred) or observed (to be conditioned on)—
    are specified by a boolean mask `condition_mask`. Dependencies among variables are
    defined by a boolean adjacency matrix `edge_mask`.

    Mask semantics: - ``condition_mask`` [B, T]: boolean per variable.
      - ``True``/1 → observed (conditioned on).
      - ``False``/0 → latent (to be inferred).
    - ``edge_mask`` [B, T, T]: boolean adjacency describing allowed attention. -
      ``True``/1 → attention from query i to key j is allowed (edge i→j). - ``False``/0
      → attention is disallowed. - If ``None``, full attention is used. Prefer passing
      ``None`` over
        an all-ones tensor to save memory.

    NOTE:
        This is the score-based implementation of the Simformer; sbi also provides a
        flow-matching variant as ``FlowMatchingSimformer``.

    NOTE:
        Multi-round inference is not supported yet.

    References:
        - [1] All-in-one simulation-based inference, Gloeckler M. et al., ICML 2024,
          https://arxiv.org/abs/2404.09636
    """

    def __init__(
        self,
        mvf_estimator: Union[
            Literal["simformer"],
            MaskedConditionalEstimatorBuilder[MaskedConditionalVectorFieldEstimator],
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
                or a callable that implements the `MaskedConditionalEstimatorBuilder`
                protocol. If a callable, `__call__` must accept `inputs`, and return
                a `MaskedConditionalVectorFieldEstimator`.
            sde_type: Type of SDE to use. Must be one of ['vp', 've', 'subvp'].
            posterior_latent_idx: List or Tensor of indices identifying which
                variables are latent (to be inferred), i.e., those that correspond
                to $\theta$ in a posterior.
            posterior_observed_idx: List or Tensor of indices identifying which
                variables are observed (to be conditioned on) in a posterior,
                i.e., those that correspond to $x$.
            device: Device to run the training on.
            logging_level: Logging level for the training. Can be an integer or a
                string.
            summary_writer: Tensorboard summary writer.
            show_progress_bars: Whether to show progress bars during training.
            kwargs: Additional keyword arguments passed to the default builder if
                `score_estimator` is a string.
        """
        super().__init__(
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

    def _build_default_nn_fn(
        self, **kwargs
    ) -> MaskedConditionalEstimatorBuilder[MaskedConditionalVectorFieldEstimator]:
        net_type = kwargs.pop("vector_field_estimator_builder", "simformer")
        return simformer_score_nn(model=net_type, **kwargs)


class FlowMatchingSimformer(MaskedVectorFieldTrainer):
    """Flow-matching version of the Simformer, Gloeckler et al. (ICML, 2024) [1].

    The flow-matching version of the Simformer shares the same architecture
    and masking semantics as the score-based Simformer above.

    See `Simformer` for details.

    NOTE:
        - Multi-round inference is not supported yet.

    References:
        - [1] All-in-one simulation-based inference, Gloeckler M. et al., ICML 2024,
          https://arxiv.org/abs/2404.09636
    """

    def __init__(
        self,
        mvf_estimator: Union[
            Literal["simformer"],
            MaskedConditionalEstimatorBuilder[MaskedConditionalVectorFieldEstimator],
        ] = "simformer",
        posterior_latent_idx: Optional[list | Tensor] = None,
        posterior_observed_idx: Optional[list | Tensor] = None,
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        **kwargs,
    ):
        r"""Initialize Flow Matching Simformer.

        Args:
            prior: Prior distribution. Its primary use is for rejecting samples that
                fall outside its defined support. For the core inference process,
                this prior is ignored.
            mvf_estimator: Neural network architecture for the masked
                vector field estimator. Can be a string (e.g., `'simformer'`)
                or a callable that implements the `MaskedConditionalEstimatorBuilder`
                protocol. If a callable, `__call__` must accept `inputs`, and return
                a `MaskedConditionalVectorFieldEstimator`.
            posterior_latent_idx: List or Tensor of indices identifying which
                variables are latent (to be inferred), i.e., those that correspond
                to $\theta$ in a posterior.
            posterior_observed_idx: List or Tensor of indices identifying which
                variables are observed (to be conditioned on) in a posterior,
                i.e., those that correspond to $x$.
            device: Device to run the training on.
            logging_level: Logging level for the training. Can be an integer or a
                string.
            summary_writer: Tensorboard summary writer.
            show_progress_bars: Whether to show progress bars during training.
            kwargs: Additional keyword arguments passed to the default builder if
                `score_estimator` is a string.
        """
        super().__init__(
            mvf_estimator_builder=mvf_estimator,
            posterior_latent_idx=posterior_latent_idx,
            posterior_observed_idx=posterior_observed_idx,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            **kwargs,
        )

    def _build_default_nn_fn(
        self, **kwargs
    ) -> MaskedConditionalEstimatorBuilder[MaskedConditionalVectorFieldEstimator]:
        model = kwargs.pop("vector_field_estimator_builder", "simformer")
        kwargs.pop("sde_type", None)  # sde_type is not used in FM Simformer
        return simformer_flow_nn(model=model, **kwargs)
