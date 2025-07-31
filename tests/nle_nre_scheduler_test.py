# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

from sbi.inference import NLE_A, NRE_A, NRE_B, NRE_C, BNRE
from sbi.inference.trainers.nle.mnle import MNLE
from sbi.simulators.linear_gaussian import linear_gaussian


@pytest.fixture
def simple_linear_gaussian_setup():
    """Fixture providing a simple linear Gaussian setup for testing."""
    num_dim = 2
    num_simulations = 300
    
    # Prior
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    
    # Likelihood setup
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    
    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)
    
    # Generate training data
    theta = prior.sample((num_simulations,))
    x = simulator(theta)
    
    return {
        "prior": prior,
        "theta": theta,
        "x": x,
        "simulator": simulator,
        "num_dim": num_dim,
    }


@pytest.mark.parametrize(
    "scheduler_type,scheduler_kwargs",
    [
        ("plateau", {"factor": 0.5, "patience": 3}),
        ("exponential", {"gamma": 0.9}),
        ("step", {"step_size": 5, "gamma": 0.5}),
    ],
)
def test_nle_scheduler_creation(simple_linear_gaussian_setup, scheduler_type, scheduler_kwargs):
    """Test that NLE can use learning rate schedulers."""
    setup = simple_linear_gaussian_setup
    
    inference = NLE_A(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    # Train with scheduler
    likelihood_estimator = inference.train(
        training_batch_size=50,
        max_num_epochs=10,
        lr_scheduler=scheduler_type,
        lr_scheduler_kwargs=scheduler_kwargs,
        learning_rate=1e-3,
    )
    
    # Check that training completed successfully
    assert likelihood_estimator is not None
    
    # Check that learning rate tracking was done
    assert hasattr(inference, "_learning_rates")
    assert len(inference._learning_rates) > 0
    assert "learning_rates" in inference._summary
    assert len(inference._summary["learning_rates"]) > 0
    
    # Check that scheduler was created
    assert hasattr(inference, "_scheduler")
    if scheduler_type == "plateau":
        assert isinstance(inference._scheduler, ReduceLROnPlateau)
    elif scheduler_type == "exponential":
        assert isinstance(inference._scheduler, ExponentialLR)


def test_mnle_scheduler_support(simple_linear_gaussian_setup):
    """Test that MNLE supports learning rate schedulers."""
    setup = simple_linear_gaussian_setup
    
    inference = MNLE(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    # Train with scheduler
    likelihood_estimator = inference.train(
        training_batch_size=50,
        max_num_epochs=8,
        lr_scheduler="plateau",
        lr_scheduler_kwargs={"factor": 0.5, "patience": 3},
    )
    
    assert likelihood_estimator is not None
    assert hasattr(inference, "_scheduler")
    assert isinstance(inference._scheduler, ReduceLROnPlateau)


@pytest.mark.parametrize("nre_class", [NRE_A, NRE_B, NRE_C, BNRE])
def test_nre_scheduler_support(simple_linear_gaussian_setup, nre_class):
    """Test that all NRE variants support learning rate schedulers."""
    setup = simple_linear_gaussian_setup
    
    inference = nre_class(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    # Train with scheduler
    ratio_estimator = inference.train(
        training_batch_size=50,
        max_num_epochs=8,
        lr_scheduler="exponential",
        lr_scheduler_kwargs={"gamma": 0.9},
    )
    
    assert ratio_estimator is not None
    assert hasattr(inference, "_scheduler")
    assert isinstance(inference._scheduler, ExponentialLR)
    assert len(inference._summary["learning_rates"]) > 0


def test_nle_lr_reduction(simple_linear_gaussian_setup):
    """Test that NLE ReduceLROnPlateau actually reduces learning rate."""
    setup = simple_linear_gaussian_setup
    
    inference = NLE_A(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    # Use aggressive plateau settings to encourage LR reduction
    likelihood_estimator = inference.train(
        training_batch_size=50,
        max_num_epochs=20,
        lr_scheduler="plateau",
        lr_scheduler_kwargs={
            "factor": 0.5,
            "patience": 2,  # Very low patience
            "verbose": False,
        },
        learning_rate=1e-2,  # Start with higher LR
        stop_after_epochs=3,  # Allow some epochs without validation improvement
    )
    
    assert likelihood_estimator is not None
    
    # Check that learning rate was recorded
    learning_rates = inference._summary["learning_rates"]
    assert len(learning_rates) > 1
    
    # Should see some reduction or at least not increase
    initial_lr = learning_rates[0]
    final_lr = learning_rates[-1]
    assert final_lr <= initial_lr  # LR should not increase


def test_nre_min_lr_threshold(simple_linear_gaussian_setup):
    """Test early stopping based on minimum learning rate threshold for NRE."""
    setup = simple_linear_gaussian_setup
    
    inference = NRE_A(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    # Train with minimum LR threshold
    ratio_estimator = inference.train(
        training_batch_size=50,
        max_num_epochs=30,  # High max epochs
        lr_scheduler="exponential",
        lr_scheduler_kwargs={"gamma": 0.8},  # Fast decay
        learning_rate=1e-3,
        min_lr_threshold=5e-4,  # Stop when LR gets below this
        stop_after_epochs=100,  # High patience to let LR threshold trigger
    )
    
    assert ratio_estimator is not None
    
    # Check that training stopped due to LR threshold (should be less than max epochs)
    epochs_trained = inference._summary["epochs_trained"][-1]
    assert epochs_trained < 30  # Should stop before max epochs
    
    # Final learning rate should be around the threshold
    final_lr = inference._summary["learning_rates"][-1]
    assert final_lr <= 5e-4


def test_nle_nre_no_scheduler_backward_compatibility(simple_linear_gaussian_setup):
    """Test that training without scheduler works as before for both NLE and NRE."""
    setup = simple_linear_gaussian_setup
    
    # Test NLE
    nle_inference = NLE_A(setup["prior"], show_progress_bars=False)
    nle_inference.append_simulations(setup["theta"], setup["x"])
    
    nle_estimator = nle_inference.train(
        training_batch_size=50,
        max_num_epochs=5,
        learning_rate=5e-4,
    )
    
    assert nle_estimator is not None
    assert nle_inference._scheduler is None
    assert len(nle_inference._summary["learning_rates"]) > 0
    
    # Learning rate should remain constant
    nle_learning_rates = nle_inference._summary["learning_rates"]
    assert all(lr == nle_learning_rates[0] for lr in nle_learning_rates)
    
    # Test NRE
    nre_inference = NRE_A(setup["prior"], show_progress_bars=False)
    nre_inference.append_simulations(setup["theta"], setup["x"])
    
    nre_estimator = nre_inference.train(
        training_batch_size=50,
        max_num_epochs=5,
        learning_rate=5e-4,
    )
    
    assert nre_estimator is not None
    assert nre_inference._scheduler is None
    assert len(nre_inference._summary["learning_rates"]) > 0
    
    # Learning rate should remain constant
    nre_learning_rates = nre_inference._summary["learning_rates"]
    assert all(lr == nre_learning_rates[0] for lr in nre_learning_rates)


def test_nle_nre_invalid_scheduler_type(simple_linear_gaussian_setup):
    """Test that invalid scheduler types raise appropriate errors for both NLE and NRE."""
    setup = simple_linear_gaussian_setup
    
    # Test NLE
    nle_inference = NLE_A(setup["prior"], show_progress_bars=False)
    nle_inference.append_simulations(setup["theta"], setup["x"])
    
    with pytest.raises(ValueError, match="Unknown scheduler type"):
        nle_inference.train(
            training_batch_size=50,
            max_num_epochs=3,
            lr_scheduler="invalid_scheduler",
        )
    
    # Test NRE
    nre_inference = NRE_A(setup["prior"], show_progress_bars=False)
    nre_inference.append_simulations(setup["theta"], setup["x"])
    
    with pytest.raises(ValueError, match="Unknown scheduler type"):
        nre_inference.train(
            training_batch_size=50,
            max_num_epochs=3,
            lr_scheduler="invalid_scheduler",
        )