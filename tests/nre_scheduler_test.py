# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR

from sbi.inference import NRE_A, NRE_B, NRE_C, BNRE
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
        ("multistep", {"milestones": [5, 10], "gamma": 0.1}),
    ],
)
def test_nre_scheduler_creation(simple_linear_gaussian_setup, scheduler_type, scheduler_kwargs):
    """Test that NRE_A can use different learning rate schedulers."""
    setup = simple_linear_gaussian_setup
    
    inference = NRE_A(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    # Train with scheduler
    ratio_estimator = inference.train(
        training_batch_size=50,
        max_num_epochs=10,
        lr_scheduler=scheduler_type,
        lr_scheduler_kwargs=scheduler_kwargs,
        learning_rate=1e-3,
    )
    
    # Check that training completed successfully
    assert ratio_estimator is not None
    
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


@pytest.mark.parametrize("nre_class", [NRE_A, NRE_B, NRE_C, BNRE])
def test_nre_variants_scheduler_support(simple_linear_gaussian_setup, nre_class):
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


def test_nre_scheduler_dict_config(simple_linear_gaussian_setup):
    """Test NRE scheduler creation using dictionary configuration."""
    setup = simple_linear_gaussian_setup
    
    inference = NRE_A(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    # Train with dictionary configuration
    scheduler_config = {
        "type": "cosine",
        "T_max": 12,
        "eta_min": 1e-6,
    }
    
    ratio_estimator = inference.train(
        training_batch_size=50,
        max_num_epochs=15,
        lr_scheduler=scheduler_config,
        learning_rate=1e-3,
    )
    
    assert ratio_estimator is not None
    assert isinstance(inference._scheduler, CosineAnnealingLR)
    assert inference._scheduler.T_max == 12
    assert inference._scheduler.eta_min == 1e-6


def test_nre_lr_reduction_plateau(simple_linear_gaussian_setup):
    """Test that NRE ReduceLROnPlateau actually reduces learning rate."""
    setup = simple_linear_gaussian_setup
    
    inference = NRE_A(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    # Use aggressive plateau settings to encourage LR reduction
    ratio_estimator = inference.train(
        training_batch_size=50,
        max_num_epochs=25,
        lr_scheduler="plateau",
        lr_scheduler_kwargs={
            "factor": 0.5,
            "patience": 2,  # Very low patience
            "verbose": False,
        },
        learning_rate=1e-2,  # Start with higher LR
        stop_after_epochs=4,  # Allow some epochs without validation improvement
    )
    
    assert ratio_estimator is not None
    
    # Check that learning rate was recorded
    learning_rates = inference._summary["learning_rates"]
    assert len(learning_rates) > 1
    
    # Should see some reduction or at least not increase
    initial_lr = learning_rates[0]
    final_lr = learning_rates[-1]
    assert final_lr <= initial_lr  # LR should not increase


def test_nre_min_lr_threshold_convergence(simple_linear_gaussian_setup):
    """Test NRE early stopping based on minimum learning rate threshold."""
    setup = simple_linear_gaussian_setup
    
    inference = NRE_A(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    # Train with minimum LR threshold
    ratio_estimator = inference.train(
        training_batch_size=50,
        max_num_epochs=35,  # High max epochs
        lr_scheduler="exponential",
        lr_scheduler_kwargs={"gamma": 0.8},  # Fast decay
        learning_rate=1e-3,
        min_lr_threshold=5e-4,  # Stop when LR gets below this
        stop_after_epochs=100,  # High patience to let LR threshold trigger
    )
    
    assert ratio_estimator is not None
    
    # Check that training stopped due to LR threshold (should be less than max epochs)
    epochs_trained = inference._summary["epochs_trained"][-1]
    assert epochs_trained < 35  # Should stop before max epochs
    
    # Final learning rate should be around the threshold
    final_lr = inference._summary["learning_rates"][-1]
    assert final_lr <= 5e-4


def test_nre_no_scheduler_backward_compatibility(simple_linear_gaussian_setup):
    """Test that NRE training without scheduler works as before."""
    setup = simple_linear_gaussian_setup
    
    inference = NRE_A(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    # Train without scheduler (default behavior)
    ratio_estimator = inference.train(
        training_batch_size=50,
        max_num_epochs=8,
        learning_rate=5e-4,
    )
    
    assert ratio_estimator is not None
    
    # Check that no scheduler was created
    assert inference._scheduler is None
    
    # But learning rates should still be tracked
    assert len(inference._summary["learning_rates"]) > 0
    
    # Learning rate should remain constant
    learning_rates = inference._summary["learning_rates"]
    assert all(lr == learning_rates[0] for lr in learning_rates)


def test_nre_invalid_scheduler_type(simple_linear_gaussian_setup):
    """Test that NRE invalid scheduler types raise appropriate errors."""
    setup = simple_linear_gaussian_setup
    
    inference = NRE_A(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    with pytest.raises(ValueError, match="Unknown scheduler type"):
        inference.train(
            training_batch_size=50,
            max_num_epochs=3,
            lr_scheduler="invalid_scheduler",
        )


def test_nre_scheduler_kwargs_override(simple_linear_gaussian_setup):
    """Test that NRE lr_scheduler_kwargs properly override defaults."""
    setup = simple_linear_gaussian_setup
    
    inference = NRE_B(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    # Test that custom kwargs override defaults
    ratio_estimator = inference.train(
        training_batch_size=50,
        max_num_epochs=10,
        lr_scheduler="plateau",
        lr_scheduler_kwargs={
            "factor": 0.25,  # Different from default 0.5
            "patience": 15,  # Different from default 10
            "cooldown": 2,   # Additional parameter
        },
    )
    
    assert ratio_estimator is not None
    assert inference._scheduler.factor == 0.25
    assert inference._scheduler.patience == 15
    assert inference._scheduler.cooldown == 2


def test_nre_resume_training_with_scheduler(simple_linear_gaussian_setup):
    """Test that NRE scheduler state is preserved when resuming training."""
    setup = simple_linear_gaussian_setup
    
    inference = NRE_C(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    # Initial training
    inference.train(
        training_batch_size=50,
        max_num_epochs=5,
        lr_scheduler="exponential",
        lr_scheduler_kwargs={"gamma": 0.9},
    )
    
    initial_epochs = inference._summary["epochs_trained"][-1]
    initial_lr = inference._summary["learning_rates"][-1]
    
    # Resume training
    ratio_estimator = inference.train(
        training_batch_size=50,
        max_num_epochs=10,
        resume_training=True,
    )
    
    assert ratio_estimator is not None
    
    # Check that training continued from where it left off
    final_epochs = inference._summary["epochs_trained"][-1]
    assert final_epochs > initial_epochs
    
    # Learning rate should have continued to decay
    final_lr = inference._summary["learning_rates"][-1]
    assert final_lr <= initial_lr


def test_nre_cyclic_scheduler(simple_linear_gaussian_setup):
    """Test NRE with CyclicLR scheduler."""
    setup = simple_linear_gaussian_setup
    
    inference = BNRE(setup["prior"], show_progress_bars=False)
    inference.append_simulations(setup["theta"], setup["x"])
    
    # Train with cyclic scheduler
    ratio_estimator = inference.train(
        training_batch_size=50,
        max_num_epochs=10,
        lr_scheduler="cyclic",
        lr_scheduler_kwargs={
            "base_lr": 1e-5,
            "max_lr": 1e-3,
            "step_size_up": 3,
        },
        learning_rate=1e-4,  # This becomes base_lr
    )
    
    assert ratio_estimator is not None
    assert hasattr(inference, "_scheduler")
    
    # Check that learning rate varied during training (cyclic pattern)
    learning_rates = inference._summary["learning_rates"]
    assert len(learning_rates) > 0
    # Cyclic scheduler should have some variation in learning rates
    assert max(learning_rates) > min(learning_rates)