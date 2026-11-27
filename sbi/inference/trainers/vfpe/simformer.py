# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import ClassVar, Literal, Optional, Union

from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.trainers.vfpe.base_vf_inference import MaskedVectorFieldTrainer
from sbi.neural_nets.estimators.base import (
    MaskedConditionalEstimatorBuilder,
    MaskedConditionalVectorFieldEstimator,
)
from sbi.neural_nets.net_builders.vector_field_nets import (
    SimformerConfigBase,
    SimformerFlowMatchingConfig,
    SimformerScoreConfigBase,
    _simformer_score_config_cls,
)
from sbi.sbi_types import Tracker


class Simformer(MaskedVectorFieldTrainer):
    """Simformer as in Gloeckler et al. (ICML, 2024) [1].

    The underlying neural network is `VectorFieldSimformer`.

    Simformer enables sampling from arbitrary conditional distributions, not just
    posterior or likelihood, by operating on a unified input tensor that represents all
    variables.

    The roles of variables—latent (to be inferred) or observed (to be conditioned on)—
    are specified by a boolean mask `condition_mask`. Dependencies among variables are
    defined by a boolean adjacency matrix `edge_mask`.

    Mask semantics:

    - ``condition_mask`` of shape ``(T,)`` (or ``(B, T)`` during training), one
      boolean per variable:

      - ``True``/1: observed (conditioned on).
      - ``False``/0: latent (to be inferred).

    - ``edge_mask`` of shape ``(T, T)`` (or ``(B, T, T)`` during training):
      ``edge_mask[i, j] = True`` means that variable ``i`` depends on variable ``j``,
      i.e., node ``i`` attends to node ``j``. For a DAG edge ``j -> i``, set
      ``edge_mask[i, j] = True``. Self-edges are always added. If ``None``, full
      attention is used; prefer ``None`` over an all-ones tensor to save memory.

    The estimator is configured with a score config, e.g.
    ``Simformer(vf_estimator=SimformerVPScoreConfig(net=SimformerNetConfig(
    num_layers=6)))``.

    NOTE:
        This is the score-based implementation of the Simformer; sbi also provides a
        flow-matching variant as ``FlowMatchingSimformer``.

    NOTE:
        Multi-round inference is not supported yet.

    References:
        - [1] All-in-one simulation-based inference, Gloeckler M. et al., ICML 2024,
          https://arxiv.org/abs/2404.09636
    """

    _ALLOWED_CONFIG_TYPE: ClassVar[type[SimformerConfigBase]] = SimformerScoreConfigBase

    def __init__(
        self,
        posterior_latent_idx: Optional[list | Tensor] = None,
        posterior_observed_idx: Optional[list | Tensor] = None,
        prior: Optional[Distribution] = None,
        vf_estimator: Union[
            SimformerScoreConfigBase,
            MaskedConditionalEstimatorBuilder[MaskedConditionalVectorFieldEstimator],
            None,
        ] = None,
        sde_type: Optional[Literal["vp", "ve", "subvp"]] = None,
        device: str = "cpu",
        logging_level: Optional[Union[int, str]] = None,
        *,
        summary_writer: None = None,
        tracker: Optional[Tracker] = None,
        show_progress_bars: bool = True,
    ):
        r"""Initialize Simformer.

        Args:
            posterior_latent_idx: List or Tensor of indices identifying which
                variables are latent (to be inferred), i.e., those that correspond
                to $\theta$ in a posterior. Must be in ascending order.
            posterior_observed_idx: List or Tensor of indices identifying which
                variables are observed (to be conditioned on) in a posterior,
                i.e., those that correspond to $x$. Must be in ascending order.
            prior: Optional distribution over the posterior latent variables. It is
                the default prior of `build_posterior` (rejection of samples outside
                its support, iid sampling). Training does not use it.
            vf_estimator: The masked score estimator. If ``None`` (default), uses
                the score config of `sde_type` with default settings
                (`SimformerVEScoreConfig`). A score config such as
                `SimformerVPScoreConfig(...)` configures the estimator and its
                network. Alternatively, a function that receives the training
                inputs and returns a `MaskedConditionalVectorFieldEstimator`.
            sde_type: Type of SDE of the default config, one of 've' (default),
                'vp' or 'subvp'. Must not be given with `vf_estimator`, which
                already selects the SDE.
            device: Device to run the training on.
            logging_level: Deprecated, it has no effect and goes in sbi v0.29.0.
                Use `logging.getLogger("sbi").setLevel(...)` instead.
            summary_writer: Removed in sbi v0.28.0; passing it raises a `TypeError`.
                Use `tracker=TensorBoardTracker(summary_writer)` instead.
            tracker: Tracking adapter used to log training metrics. If None, a
                TensorBoard tracker is used with a default log directory.
            show_progress_bars: Whether to show progress bars during training.
        """
        if vf_estimator is None:
            vf_estimator = _simformer_score_config_cls(sde_type or "ve")()
        elif sde_type is not None:
            raise ValueError(
                f"Conflicting `sde_type`: trainer received {sde_type!r}, but "
                "`vf_estimator` already selects the SDE. Choose the config of the "
                "SDE instead, e.g. `SimformerVPScoreConfig()`."
            )
        super().__init__(
            vf_estimator=vf_estimator,
            posterior_latent_idx=posterior_latent_idx,
            posterior_observed_idx=posterior_observed_idx,
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            tracker=tracker,
            show_progress_bars=show_progress_bars,
        )


class FlowMatchingSimformer(MaskedVectorFieldTrainer):
    """Flow-matching version of the Simformer, Gloeckler et al. (ICML, 2024) [1].

    The underlying neural network is `VectorFieldSimformer`.

    The flow-matching version of the Simformer shares the same architecture
    and masking semantics as the score-based Simformer above.

    See `Simformer` for details.

    NOTE:
        - Multi-round inference is not supported yet.

    References:
        - [1] All-in-one simulation-based inference, Gloeckler M. et al., ICML 2024,
          https://arxiv.org/abs/2404.09636
    """

    _ALLOWED_CONFIG_TYPE: ClassVar[type[SimformerConfigBase]] = (
        SimformerFlowMatchingConfig
    )
    _DEFAULT_SAMPLE_WITH: ClassVar[Literal["ode", "sde"]] = "ode"

    def __init__(
        self,
        posterior_latent_idx: Optional[list | Tensor] = None,
        posterior_observed_idx: Optional[list | Tensor] = None,
        prior: Optional[Distribution] = None,
        vf_estimator: Union[
            SimformerFlowMatchingConfig,
            MaskedConditionalEstimatorBuilder[MaskedConditionalVectorFieldEstimator],
            None,
        ] = None,
        device: str = "cpu",
        logging_level: Optional[Union[int, str]] = None,
        *,
        summary_writer: None = None,
        tracker: Optional[Tracker] = None,
        show_progress_bars: bool = True,
    ):
        r"""Initialize Flow Matching Simformer.

        Args:
            posterior_latent_idx: List or Tensor of indices identifying which
                variables are latent (to be inferred), i.e., those that correspond
                to $\theta$ in a posterior. Must be in ascending order.
            posterior_observed_idx: List or Tensor of indices identifying which
                variables are observed (to be conditioned on) in a posterior,
                i.e., those that correspond to $x$. Must be in ascending order.
            prior: Optional distribution over the posterior latent variables. It is
                the default prior of `build_posterior` (rejection of samples outside
                its support, iid sampling). Training does not use it.
            vf_estimator: The masked flow-matching estimator. If ``None``
                (default), uses `SimformerFlowMatchingConfig()`. A config
                configures the estimator and its network. Alternatively, a function
                that receives the training inputs and returns a
                `MaskedConditionalVectorFieldEstimator`.
            device: Device to run the training on.
            logging_level: Deprecated, it has no effect and goes in sbi v0.29.0.
                Use `logging.getLogger("sbi").setLevel(...)` instead.
            summary_writer: Removed in sbi v0.28.0; passing it raises a `TypeError`.
                Use `tracker=TensorBoardTracker(summary_writer)` instead.
            tracker: Tracking adapter used to log training metrics. If None, a
                TensorBoard tracker is used with a default log directory.
            show_progress_bars: Whether to show progress bars during training.
        """
        super().__init__(
            vf_estimator=(
                SimformerFlowMatchingConfig() if vf_estimator is None else vf_estimator
            ),
            posterior_latent_idx=posterior_latent_idx,
            posterior_observed_idx=posterior_observed_idx,
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            tracker=tracker,
            show_progress_bars=show_progress_bars,
        )
