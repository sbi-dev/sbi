# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Dict, Literal, Optional, Union

from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.trainers.npe.npe_base import (
    PosteriorEstimatorTrainer,
)
from sbi.inference.trainers.npe.npe_loss import (
    ImportanceWeightedLoss,
    NPELossStrategy,
)
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimatorBuilder,
)
from sbi.sbi_types import Tracker
from sbi.utils.sbiutils import del_entries


class NPE_B(PosteriorEstimatorTrainer):
    r"""Neural Posterior Estimation algorithm (NPE-B) as in Lueckmann et al. (2017) [1].

    NPE-B (also known as SNPE-B) trains a neural network to directly approximate
    the posterior $p(\theta|x)$ using an importance-weighted loss. Unlike NPE-A,
    this importance weighting ensures convergence to the true posterior in multi-round
    inference, and it is not limited to Gaussian proposals. NPE-B can use flexible
    density estimators like normalizing flows.

    For single-round inference, NPE-A, NPE-B, and NPE-C are equivalent and use
    plain NLL loss.

    [1] *Flexible statistical inference for mechanistic models of neural
        dynamics*, Lueckmann, Gonçalves et al., NeurIPS 2017.
        https://arxiv.org/abs/1711.01861

    Example:
    --------

    ::

        import torch
        from sbi.inference import NPE_B
        from sbi.utils import BoxUniform

        # 1. Setup simulator, prior, and observation
        prior = BoxUniform(low=torch.zeros(3), high=torch.ones(3))
        x_o = torch.randn(1, 3)  # Observed data

        def simulator(theta):
            return theta + torch.randn_like(theta) * 0.1

        # 2. Multi-round inference
        inference = NPE_B(prior=prior)
        proposal = prior

        for round_idx in range(5):
            theta = proposal.sample((100,))
            x = simulator(theta)
            density_estimator = inference.append_simulations(theta, x).train()
            posterior = inference.build_posterior(density_estimator)
            proposal = posterior.set_default_x(x_o)

        # 3. Sample from final posterior
        samples = posterior.sample((1000,), x=x_o)
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[
            Literal["nsf", "maf", "mdn", "made"],
            ConditionalEstimatorBuilder[ConditionalDensityEstimator],
        ] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        tracker: Optional[Tracker] = None,
        show_progress_bars: bool = True,
    ):
        r"""Initialize NPE-B.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network, which adheres to
                `ConditionalEstimatorBuilder` protocol can be provided. The function
                will be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. The
                density estimator needs to provide the methods `.log_prob` and
                `.sample()` and must return a `ConditionalDensityEstimator`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: Deprecated alias for the TensorBoard summary writer.
                Use ``tracker`` instead.
            tracker: Tracking adapter used to log training metrics. If None, a
                TensorBoard tracker is used with a default log directory.
            show_progress_bars: Whether to show a progressbar during training.
        """

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
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
        loss_strategy: Optional[NPELossStrategy] = None,
    ) -> ConditionalDensityEstimator:

        kwargs = del_entries(
            locals(),
            entries=("self", "__class__", "loss_strategy"),
        )

        if len(self._data_round_index) == 0:
            raise RuntimeError(
                "No simulations found. You must call .append_simulations() "
                "before calling .train()."
            )

        self._round = max(self._data_round_index)

        if loss_strategy is not None:
            self._loss_strategy = loss_strategy
        elif self._round > 0:
            self._loss_strategy = ImportanceWeightedLoss(
                neural_net=self._neural_net,
                prior=self._prior,
                round_idx=self._round,
                theta_roundwise=self._theta_roundwise,
                proposal_roundwise=self._proposal_roundwise,
            )
        else:
            self._loss_strategy = None

        return super().train(**kwargs)
