from dataclasses import asdict

import pytest
import torch

from sbi.inference import NPE, NRE
from sbi.inference.trainers._contracts import (
    LossArgsBNRE,
    LossArgsNPE,
    LossArgsNRE,
    LossArgsNRE_A,
    LossArgsNRE_C,
    LossArgsVF,
    StartIndexContext,
    TrainConfig,
)
from sbi.inference.trainers.nre.bnre import BNRE
from sbi.inference.trainers.nre.nre_a import NRE_A
from sbi.inference.trainers.nre.nre_c import NRE_C
from sbi.inference.trainers.vfpe.fmpe import FMPE
from sbi.utils.torchutils import BoxUniform

non_positive_values = [-1, 0, False]


@pytest.fixture()
def train_config_dict():
    """Test fixture for training dictionary hyperparameters."""

    return dict(
        training_batch_size=1,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=20,
        max_num_epochs=1,
        resume_training=False,
        retrain_from_scratch=False,
        show_train_summary=False,
    )


@pytest.mark.parametrize(
    ("inference_class", "loss_args_dataclass", "loss_args", "skipped_argument"),
    [
        (NRE, LossArgsNRE, dict(num_atoms=9), None),
        (NRE_A, LossArgsNRE_A, dict(), "num_atoms"),
        (NRE_C, LossArgsNRE_C, dict(gamma=1.0), "num_atoms"),
        (BNRE, LossArgsBNRE, dict(regularization_strength=100.0), "num_atoms"),
        (NPE, LossArgsNPE, dict(force_first_round_loss=False), "proposal"),
        (FMPE, LossArgsVF, dict(force_first_round_loss=False), "proposal"),
        # Failing cases
        *[
            pytest.param(
                NRE,
                LossArgsNRE,
                dict(num_atoms=value),
                None,
                marks=pytest.mark.xfail(raises=(TypeError, ValueError)),
            )
            for value in non_positive_values
        ],
        *[
            pytest.param(
                NPE,
                LossArgsNRE_C,
                dict(gamma=value),
                None,
                marks=pytest.mark.xfail(raises=(TypeError, ValueError)),
            )
            for value in non_positive_values
        ],
        *[
            pytest.param(
                BNRE,
                LossArgsBNRE,
                dict(regularization_strength=value),
                None,
                marks=pytest.mark.xfail(raises=(TypeError, ValueError)),
            )
            for value in non_positive_values
        ],
    ],
)
def test_trainer_dataclass_loss_args_validation(
    inference_class, loss_args_dataclass, loss_args, skipped_argument
):
    def simulator(theta):
        return 1.0 + theta + torch.randn(theta.shape, device=theta.device) * 0.1

    num_dim = 3
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
    theta = prior.sample((300,))
    x = simulator(theta)

    inference = inference_class(prior=prior)
    inference.append_simulations(theta, x)

    dataclass = loss_args_dataclass(**loss_args)

    kwargs = asdict(dataclass)

    if skipped_argument is not None:
        kwargs.pop(skipped_argument)

    if isinstance(dataclass, LossArgsVF):
        kwargs["validation_times"] = kwargs.pop("times")

    inference.train(**kwargs)


@pytest.mark.parametrize(
    ("train_config_args"),
    [
        {},
        # Failing cases
        *[
            pytest.param(
                dict(training_batch_size=value),
                marks=pytest.mark.xfail(raises=(TypeError, ValueError)),
            )
            for value in non_positive_values
        ],
        *[
            pytest.param(
                dict(learning_rate=value),
                marks=pytest.mark.xfail(raises=(TypeError, ValueError)),
            )
            for value in non_positive_values
        ],
        *[
            pytest.param(
                dict(validation_fraction=value),
                marks=pytest.mark.xfail(raises=(TypeError, ValueError)),
            )
            for value in non_positive_values + [1, 1.5]
        ],
        *[
            pytest.param(
                dict(stop_after_epochs=value),
                marks=pytest.mark.xfail(raises=(TypeError, ValueError)),
            )
            for value in non_positive_values
        ],
        *[
            pytest.param(
                dict(max_num_epochs=value),
                marks=pytest.mark.xfail(raises=(TypeError, ValueError)),
            )
            for value in non_positive_values
        ],
        *[
            pytest.param(
                dict(clip_max_norm=value),
                marks=pytest.mark.xfail(raises=(TypeError, ValueError)),
            )
            for value in non_positive_values
        ],
    ],
)
def test_trainer_config_validation(train_config_args, train_config_dict):
    def simulator(theta):
        return 1.0 + theta + torch.randn(theta.shape, device=theta.device) * 0.1

    num_dim = 3
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
    theta = prior.sample((300,))
    x = simulator(theta)

    inference = NPE(prior=prior)
    inference.append_simulations(theta, x)

    train_config_dict = {**train_config_dict, **train_config_args}

    train_config = TrainConfig(**train_config_dict)

    inference.train(**asdict(train_config))


@pytest.mark.parametrize(
    ("dataclass_args"),
    [
        dict(
            discard_prior_samples=True,
            resume_training=False,
            force_first_round_loss=False,
        ),
    ],
)
def test_trainer_dataclass_start_index_context_validation(dataclass_args):
    def simulator(theta):
        return 1.0 + theta + torch.randn(theta.shape, device=theta.device) * 0.1

    num_dim = 3
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
    theta = prior.sample((300,))
    x = simulator(theta)

    inference = NPE(prior=prior)
    inference.append_simulations(theta, x)

    dataclass = StartIndexContext(**dataclass_args)

    kwargs = asdict(dataclass)

    inference.train(**kwargs)


@pytest.mark.parametrize(
    "loss_kwargs",
    [
        pytest.param({}, marks=pytest.mark.xfail(raises=TypeError)),
        pytest.param(LossArgsNRE(), marks=pytest.mark.xfail(raises=TypeError)),
    ],
)
def test_nre_a_train_method_invalid_argument_raises_error(loss_kwargs):
    """
    This test checks if an error is raised when an incorrect type is passed
    to loss_kwargs in NRE_A.
    """

    def simulator(theta):
        return 1.0 + theta + torch.randn(theta.shape, device=theta.device) * 0.1

    num_dim = 3
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
    theta = prior.sample((300,))
    x = simulator(theta)

    inference = NRE_A(prior=prior)
    inference.append_simulations(theta, x)

    inference.train(loss_kwargs=loss_kwargs)
