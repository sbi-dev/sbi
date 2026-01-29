# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import pytest
import torch

from sbi.inference import NPE, NRE
from sbi.inference.trainers._contracts import (
    LossArgsNRE,
)
from sbi.inference.trainers.nre.bnre import BNRE
from sbi.inference.trainers.nre.nre_a import NRE_A
from sbi.inference.trainers.nre.nre_c import NRE_C
from sbi.inference.trainers.vfpe.fmpe import FMPE
from sbi.utils.torchutils import BoxUniform

non_positive_values = [-1, 0, False]


@pytest.mark.parametrize(
    ("inference_class", "loss_args"),
    [
        (NRE, dict(num_atoms=10)),
        (NRE_C, dict(gamma=1.0)),
        (BNRE, dict(regularization_strength=100.0)),
        (NPE, dict(force_first_round_loss=False)),
        (FMPE, dict(force_first_round_loss=False)),
        # Failing cases
        *[
            pytest.param(
                NRE,
                dict(num_atoms=value),
                marks=pytest.mark.xfail(raises=(TypeError, ValueError)),
            )
            for value in non_positive_values
        ],
        *[
            pytest.param(
                NPE,
                dict(gamma=value),
                marks=pytest.mark.xfail(raises=(TypeError, ValueError)),
            )
            for value in non_positive_values
        ],
        *[
            pytest.param(
                BNRE,
                dict(regularization_strength=value),
                marks=pytest.mark.xfail(raises=(TypeError, ValueError)),
            )
            for value in non_positive_values
        ],
    ],
)
def test_loss_args_dataclass_validation(inference_class, loss_args):
    """
    Test train method loss arguments that pass through LossArgs dataclasses are
    validated.
    """

    def simulator(theta):
        return 1.0 + theta + torch.randn(theta.shape, device=theta.device) * 0.1

    num_dim = 3
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
    theta = prior.sample((300,))
    x = simulator(theta)

    inference = inference_class(prior=prior)
    inference.append_simulations(theta, x)

    inference.train(**loss_args)


@pytest.mark.parametrize(
    ("train_config_args"),
    [
        dict(
            training_batch_size=1,
            learning_rate=5e-4,
            validation_fraction=0.1,
            stop_after_epochs=20,
            max_num_epochs=1,
            resume_training=False,
            retrain_from_scratch=False,
            show_train_summary=False,
        ),
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
def test_trainer_config_dataclass_validation(train_config_args):
    """
    Test train method arguments that pass through TrainConfig dataclass are
    validated.
    """

    def simulator(theta):
        return 1.0 + theta + torch.randn(theta.shape, device=theta.device) * 0.1

    num_dim = 3
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
    theta = prior.sample((300,))
    x = simulator(theta)

    inference = NPE(prior=prior)
    inference.append_simulations(theta, x)

    inference.train(**train_config_args)


@pytest.mark.parametrize(
    ("dataclass_args"),
    [
        dict(
            discard_prior_samples=True,
            resume_training=False,
            force_first_round_loss=False,
        ),
        pytest.param(
            dict(discard_prior_samples=None),
            marks=pytest.mark.xfail(raises=(TypeError)),
        ),
    ],
)
def test_trainer_dataclass_start_index_context_validation(dataclass_args):
    """
    Test train method arguments that pass through StartIndexContext dataclass are
    validated.
    """

    def simulator(theta):
        return 1.0 + theta + torch.randn(theta.shape, device=theta.device) * 0.1

    num_dim = 3
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
    theta = prior.sample((300,))
    x = simulator(theta)

    inference = NPE(prior=prior)
    inference.append_simulations(theta, x)

    inference.train(**dataclass_args)


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
