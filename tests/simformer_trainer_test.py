# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Trainer-level tests of the Simformer: priors, mask generators, edge masks of
conditionals, pickling and invalid entries."""

import pickle

import pytest
import torch
from torch.distributions import MultivariateNormal

from sbi.inference import FlowMatchingSimformer, Simformer
from sbi.inference.trainers.base import infer
from sbi.neural_nets.estimators.base import (
    MaskedConditionalVectorFieldEstimatorWrapper,
)
from sbi.neural_nets.factory import simformer_flow_nn, simformer_score_nn
from sbi.utils import BoxUniform
from sbi.utils.sbiutils import handle_invalid_inputs_for_simformer

NUM_NODES = 3
CHAIN_DAG = torch.tensor([
    [True, False, False],
    [True, True, False],
    [False, True, True],
])


def _unbatched_condition_mask(inputs):
    """Module-level generator returning one `(T,)` mask for the whole batch."""
    return torch.tensor([False, True, True])


def _unbatched_chain_edge_mask(condition_mask):
    """Module-level generator returning one `(T, T)` mask for the whole batch."""
    return CHAIN_DAG


def _trained(method=Simformer, num_epochs=2, **kwargs):
    inference = method(
        posterior_latent_idx=[0], posterior_observed_idx=[1, 2], **kwargs
    )
    inference.append_simulations(torch.randn(100, NUM_NODES))
    inference.train(max_num_epochs=num_epochs, validation_fraction=0.2)
    return inference


def _wrapper(posterior) -> MaskedConditionalVectorFieldEstimatorWrapper:
    estimator = posterior.vector_field_estimator
    assert isinstance(estimator, MaskedConditionalVectorFieldEstimatorWrapper)
    return estimator


@pytest.mark.parametrize("method", [Simformer, FlowMatchingSimformer])
def test_prior_passed_at_init_is_the_default_posterior_prior(method):
    prior = BoxUniform(-torch.ones(1), torch.ones(1))
    inference = _trained(method, prior=prior)
    assert inference._prior is prior
    assert inference.build_posterior().prior is prior
    # The prior is over the posterior latents, so it is not a likelihood prior.
    assert inference.build_likelihood().prior is None


def test_fixed_and_listed_masks_keep_the_trainer_picklable():
    inference = Simformer(posterior_latent_idx=[0], posterior_observed_idx=[1, 2])
    inference.set_condition_masks_generator([
        torch.tensor([False, True, True]),
        torch.tensor([True, False, False]),
    ])
    inference.set_edge_masks_generator(CHAIN_DAG)
    inference.append_simulations(torch.randn(100, NUM_NODES))
    inference.train(max_num_epochs=2)
    restored = pickle.loads(pickle.dumps(inference))
    assert torch.equal(restored._edge_mask_generator.mask, CHAIN_DAG)


def test_mask_set_is_drawn_in_a_fixed_order():
    masks = [torch.tensor([i % 2 == 0, i % 3 == 0, True]) for i in range(5)]
    draws = []
    for order in (masks, masks[::-1]):
        inference = Simformer()
        inference.set_condition_masks_generator(set(order))
        torch.manual_seed(0)
        draws.append(inference._condition_mask_generator(torch.zeros(20, NUM_NODES)))
    assert torch.equal(draws[0], draws[1])


def test_mask_generator_rejects_unsupported_types():
    with pytest.raises(TypeError, match="condition_mask_generator"):
        Simformer().set_condition_masks_generator("posterior")


def test_unbatched_generator_masks_train_and_validate():
    inference = Simformer(posterior_latent_idx=[0], posterior_observed_idx=[1, 2])
    inference.set_condition_masks_generator(_unbatched_condition_mask)
    inference.set_edge_masks_generator(_unbatched_chain_edge_mask)
    inference.append_simulations(torch.randn(100, NUM_NODES))
    inference.train(max_num_epochs=2)
    validation_losses = torch.tensor(inference.summary["validation_loss"])
    assert len(validation_losses) > 0 and torch.isfinite(validation_losses).all()


def test_generator_masks_of_wrong_shape_raise():
    inference = Simformer(posterior_latent_idx=[0], posterior_observed_idx=[1, 2])
    inference.set_edge_masks_generator(lambda condition_mask: torch.ones(2, 2))
    inference.append_simulations(torch.randn(100, NUM_NODES))
    with pytest.raises(ValueError, match="edge mask generator must return"):
        inference.train(max_num_epochs=1)


def test_conditionals_use_the_trained_edge_mask():
    inference = Simformer(posterior_latent_idx=[0], posterior_observed_idx=[1, 2])
    inference.set_edge_masks_generator(CHAIN_DAG)
    inference.append_simulations(torch.randn(100, NUM_NODES))
    inference.train(max_num_epochs=2)

    for posterior in (inference.build_posterior(), inference.build_likelihood()):
        assert torch.equal(_wrapper(posterior)._fixed_edge_mask, CHAIN_DAG)
    # An explicit edge mask still wins.
    full = torch.ones(NUM_NODES, NUM_NODES, dtype=torch.bool)
    posterior = inference.build_posterior(edge_mask=full)
    assert torch.equal(_wrapper(posterior)._fixed_edge_mask, full)


def test_conditionals_use_a_condition_dependent_edge_mask():
    def edge_mask_fn(condition_mask):
        # Latent nodes do not attend to the observed node 2.
        mask = torch.ones(condition_mask.shape[0], NUM_NODES, NUM_NODES).bool()
        mask[:, :, 2] = condition_mask[:, 2:3] == 0
        return mask

    inference = _trained()
    inference.set_edge_masks_generator(edge_mask_fn)
    edge_mask = _wrapper(inference.build_posterior())._fixed_edge_mask
    assert edge_mask is not None and not edge_mask[:, 2].any()


def test_conditionals_with_random_edge_masks_need_an_explicit_edge_mask():
    inference = _trained()
    inference.set_edge_masks_generator([
        CHAIN_DAG,
        torch.ones(NUM_NODES, NUM_NODES, dtype=torch.bool),
    ])
    with pytest.raises(ValueError, match="Pass `edge_mask` explicitly"):
        inference.build_posterior()
    inference.build_posterior(edge_mask=CHAIN_DAG)


def test_flow_factory_honours_z_score_x():
    inputs = 5.0 + 3.0 * torch.randn(200, NUM_NODES)
    estimator = simformer_flow_nn(z_score_x=None)(inputs)
    assert torch.all(estimator.mean_0 == 0) and torch.all(estimator.std_0 == 1)
    estimator = simformer_flow_nn()(inputs)
    assert torch.allclose(estimator.mean_0, torch.full((NUM_NODES,), 5.0), atol=1)


@pytest.mark.parametrize("factory", [simformer_score_nn, simformer_flow_nn])
def test_factories_reject_an_embedding_net(factory):
    with pytest.raises(ValueError, match="embedding_net"):
        factory(embedding_net=torch.nn.Linear(1, 1))


@pytest.mark.parametrize("num_features", [None, 2])
@pytest.mark.parametrize("method", [Simformer, FlowMatchingSimformer])
def test_wrapper_time_statistics_have_one_trailing_dim(method, num_features):
    shape = (100, NUM_NODES) if num_features is None else (100, NUM_NODES, 2)
    inference = method(posterior_latent_idx=[0], posterior_observed_idx=[1, 2])
    inference.append_simulations(torch.randn(shape))
    inference.train(max_num_epochs=1)
    wrapper = _wrapper(inference.build_posterior())
    times = torch.rand(4)
    for statistic in (wrapper.mean_t_fn(times), wrapper.std_fn(times)):
        assert statistic.shape[:-1] == (4,)
        assert statistic.shape[-1] in (1, wrapper.input_shape[0])


@pytest.mark.gpu
def test_condition_mask_from_indices_on_the_gpu():
    inference = Simformer(
        posterior_latent_idx=torch.tensor([0], device="cuda"),
        posterior_observed_idx=torch.tensor([1, 2], device="cuda"),
        device="cuda",
    )
    condition_mask = inference._generate_posterior_condition_mask()
    assert condition_mask.device.type == "cuda"
    assert condition_mask.tolist() == [False, True, True]


def test_infer_runs_the_simformer():
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    posterior = infer(
        lambda theta: theta + 0.1 * torch.randn_like(theta),
        prior,
        method="simformer",
        num_simulations=100,
        train_kwargs=dict(max_num_epochs=2),
    )
    assert posterior.sample((5,), x=torch.zeros(2)).shape == (5, 2)


def test_invalid_entries_are_latent_and_excluded_from_the_loss():
    inputs = torch.randn(4, NUM_NODES, 2)
    inputs[0, 1, 0] = float("nan")
    inputs[1, 2, 1] = float("inf")
    condition_masks = torch.ones(4, NUM_NODES, dtype=torch.bool)
    cleaned, condition_masks, valid = handle_invalid_inputs_for_simformer(
        inputs, condition_masks
    )
    assert torch.isfinite(cleaned).all()
    assert not condition_masks[0, 1] and not condition_masks[1, 2]
    assert condition_masks.sum() == 4 * NUM_NODES - 2
    assert valid.sum() == inputs.numel() - 2

    with pytest.raises(ValueError, match="No valid data entries"):
        handle_invalid_inputs_for_simformer(torch.full((2, 3), float("nan")))


@pytest.mark.parametrize("method", [Simformer, FlowMatchingSimformer])
def test_loss_ignores_entries_outside_the_loss_mask(method):
    inference = _trained(method, num_epochs=1)
    estimator = inference._neural_net
    inputs = torch.randn(8, NUM_NODES)
    condition_mask = torch.zeros(8, NUM_NODES, dtype=torch.bool)
    times = torch.full((8,), 0.01)
    loss = estimator.loss(
        inputs,
        condition_mask,
        None,
        times=times,
        loss_mask=torch.zeros_like(inputs, dtype=torch.bool),
    )
    assert torch.all(loss == 0)
