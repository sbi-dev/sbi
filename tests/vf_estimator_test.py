# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from sbi.neural_nets import (
    FlowMatchingConfig,
    SubVPScoreConfig,
    VEScoreConfig,
    VPScoreConfig,
    VectorFieldConfigBase,
)
from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.neural_nets.net_builders import (
    build_masked_flow_matching_estimator,
    build_masked_score_matching_estimator,
)
from sbi.utils.torchutils import process_device

SCORE_CONFIGS = [VPScoreConfig, SubVPScoreConfig, VEScoreConfig]
CONFIGS = [FlowMatchingConfig, *SCORE_CONFIGS]

# *** ======== Standard VF Estimators ======== *** #


@pytest.mark.parametrize("input_sample_dim", (1, 2, 3))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,), (3, 3)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("config_cls", CONFIGS)
def test_vector_field_estimator_loss_shapes(
    input_sample_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
    config_cls,
):
    """Test whether `loss` of vector field estimators follow the shape convention."""
    estimator, inputs, conditions = _build_estimator_and_tensors(
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
        config_cls=config_cls,
    )

    losses = estimator.loss(inputs[0], condition=conditions)  # type: ignore

    assert losses.shape == (batch_dim,)


@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("config_cls", SCORE_CONFIGS)
def test_score_loss_is_invariant_to_theta_dimensions(config_cls, input_event_shape):
    """Test that the score loss does not scale with the number of theta dimensions.

    With the network output pinned to a constant ``c``, the eps-dependent terms of the
    loss cancel exactly against the control variate, so the weighted loss reduces to
    ``weights * (c**2 + 1 / std**2)`` for any noise draw and number of parameters.
    """
    score_value = 0.5
    batch_dim = 4
    condition_event_shape = (1,)

    estimator = config_cls().build(
        torch.randn(batch_dim, *input_event_shape),
        torch.randn(batch_dim, *condition_event_shape),
    )
    estimator.forward = lambda input, condition, time: torch.full_like(
        input, score_value
    )

    input = torch.randn(batch_dim, *input_event_shape)
    condition = torch.randn(batch_dim, *condition_event_shape)
    times = torch.full((batch_dim,), 0.5)

    loss = estimator.loss(
        input, condition, times=times, control_variate_threshold=float("inf")
    )

    std = torch.squeeze(estimator.std_fn(times), -1)
    expected = estimator.weight_fn(times) * (score_value**2 + 1.0 / std**2)

    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("weight_fn", ("identity", "max_likelihood", "variance"))
@pytest.mark.parametrize("config_cls", SCORE_CONFIGS)
def test_score_loss_weights_each_sample(config_cls, weight_fn):
    """Test that every time weighting returns one weighted loss per sample."""
    batch_dim = 4
    estimator = config_cls(extra_kwargs={"weight_fn": weight_fn}).build(
        torch.randn(batch_dim, 3), torch.randn(batch_dim, 2)
    )

    loss = estimator.loss(torch.randn(batch_dim, 3), torch.randn(batch_dim, 2))

    assert loss.shape == (batch_dim,)


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("config_cls", CONFIGS)
def test_vector_field_estimator_on_device(device, config_cls):
    """Test whether vector field estimators can be moved to the device."""
    device = process_device(device)
    estimator = config_cls().build(torch.randn(100, 1), torch.randn(100, 1))
    estimator.to(device)

    # Test forward
    inputs = torch.randn(100, 1, device=device)
    condition = torch.randn(100, 1, device=device)
    time = torch.randn(1, device=device)
    out = estimator(inputs, condition, time)

    assert out.device.type == device.split(":")[0], "Output device mismatch."

    # Test loss
    loss = estimator.loss(inputs, condition)
    assert loss.device.type == device.split(":")[0], "Loss device mismatch."


@pytest.mark.parametrize("input_sample_dim", (1, 2, 3))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,), (3, 3)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("config_cls", CONFIGS)
def test_vector_field_estimator_forward_shapes(
    input_sample_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
    config_cls,
):
    """Test whether `forward` of vector field estimators follow the shape convention."""
    estimator, inputs, conditions = _build_estimator_and_tensors(
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
        config_cls=config_cls,
    )
    # Batched times
    times = torch.rand((batch_dim,))
    outputs = estimator(inputs[0], condition=conditions, time=times)
    assert outputs.shape == (batch_dim, *input_event_shape), "Output shape mismatch."

    # Single time
    time = torch.rand(())
    outputs = estimator(inputs[0], condition=conditions, time=time)
    assert outputs.shape == (batch_dim, *input_event_shape), "Output shape mismatch."


def _build_estimator_and_tensors(
    input_event_shape: Tuple[int],
    condition_event_shape: Tuple[int],
    batch_dim: int,
    input_sample_dim: int = 1,
    config_cls: type[VectorFieldConfigBase] = FlowMatchingConfig,
):
    """
    Helper function for all tests that deal with shapes of vector field estimators.
    """
    # Use discrete thetas such that categorical density estimators can also use them.
    building_thetas = torch.randint(
        0, 4, (100, *input_event_shape), dtype=torch.float32
    )
    building_xs = torch.randn((100, *condition_event_shape))

    if len(condition_event_shape) > 1:
        embedding_net = CNNEmbedding(condition_event_shape, kernel_size=1)
    else:
        embedding_net = torch.nn.Identity()

    estimator = config_cls(embedding_net=embedding_net).build(
        torch.randn_like(building_thetas), torch.randn_like(building_xs)
    )

    inputs = building_thetas[:batch_dim]
    condition = building_xs[:batch_dim]

    inputs = inputs.unsqueeze(0)
    inputs = inputs.expand(
        [
            input_sample_dim,
        ]
        + [-1] * (1 + len(input_event_shape))
    )
    condition = condition
    return estimator, inputs, condition


# *** ======== Masked Estimator ======== *** #


@pytest.mark.parametrize("sde_type", ["ve", "vp", "subvp", "flow"])
@pytest.mark.parametrize("input_sample_dim", (1, 2, 3))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,), (3, 5), (3, 1)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("net", ["simformer"])
def test_masked_vector_field_estimator_loss_shapes(
    sde_type,
    input_sample_dim,
    input_event_shape,
    batch_dim,
    net,
):
    """Test whether `loss` of MaskedScoreEstimator follows the shape convention."""
    (
        score_estimator,
        inputs,
        condition_masks,
        edge_masks,
    ) = _build_masked_vector_field_estimator_and_tensors(
        sde_type,
        input_event_shape,
        batch_dim,
        input_sample_dim,
        net=net,
    )

    losses = score_estimator.loss(
        inputs[0], condition_mask=condition_masks, edge_mask=edge_masks
    )
    assert losses.shape == (batch_dim,), "Loss shape mismatch."


@pytest.mark.parametrize("sde_type", ["ve", "vp"])
@pytest.mark.parametrize("num_features", [None, 1, 3])
def test_masked_score_control_variate_reduces_variance(sde_type, num_features):
    """The control variate must keep the masked score loss unbiased and reduce its
    variance at small diffusion times, for [B, T], [B, T, 1] and [B, T, F] inputs."""
    torch.manual_seed(0)
    num_nodes, batch_dim = 6, 64
    shape = (num_nodes,) if num_features is None else (num_nodes, num_features)
    estimator = build_masked_score_matching_estimator(
        torch.randn(100, *shape), torch.randn(100, *shape), sde_type=sde_type
    )
    estimator.eval()
    inputs = torch.randn(batch_dim, *shape)
    condition_mask = torch.zeros(batch_dim, num_nodes, dtype=torch.bool)
    condition_mask[:, num_nodes // 2 :] = True
    # Small time, so that std_t is below the control variate threshold.
    times = torch.full((batch_dim,), 0.05)
    assert (estimator.std_fn(times) < 0.3).all()

    losses = {}
    with torch.no_grad():
        for control_variate in (False, True):
            losses[control_variate] = torch.stack([
                estimator.loss(
                    inputs,
                    condition_mask,
                    None,
                    times=times,
                    control_variate=control_variate,
                )
                for _ in range(32)
            ])  # [num_repeats, batch_dim]

    var_plain = losses[False].var(dim=0).mean()
    var_cv = losses[True].var(dim=0).mean()
    assert var_cv < 0.1 * var_plain, (
        f"Control variate did not reduce variance: {var_cv:.3e} vs {var_plain:.3e}."
    )
    # Unbiased: averages over noise draws agree up to Monte Carlo error.
    mean_plain, mean_cv = losses[False].mean(), losses[True].mean()
    se = (var_plain / (32 * batch_dim)).sqrt()
    assert (mean_plain - mean_cv).abs() < 5 * se


@pytest.mark.parametrize("sde_type", ["ve", "vp", "subvp"])
@pytest.mark.parametrize("control_variate", [False, True])
def test_masked_score_loss_same_for_2d_and_trailing_feature_inputs(
    sde_type, control_variate
):
    """[B, T] and [B, T, 1] inputs describe the same data, so with the same weights
    and the same noise they must give the same loss (with or without the control
    variate)."""
    torch.manual_seed(0)
    num_nodes, batch_dim = 6, 16
    data = torch.randn(100, num_nodes)
    estimator_2d = build_masked_score_matching_estimator(
        data, data, sde_type=sde_type
    ).eval()
    estimator_3d = build_masked_score_matching_estimator(
        data.unsqueeze(-1), data.unsqueeze(-1), sde_type=sde_type
    ).eval()
    # Share the weights; only the z-scoring buffers differ in shape.
    target_state = estimator_3d.state_dict()
    estimator_3d.load_state_dict({
        key: value.reshape(target_state[key].shape)
        for key, value in estimator_2d.state_dict().items()
    })

    inputs = torch.randn(batch_dim, num_nodes)
    condition_mask = torch.zeros(batch_dim, num_nodes, dtype=torch.bool)
    condition_mask[:, num_nodes // 2 :] = True
    # Small times, so that the control variate is active.
    times = torch.rand(batch_dim) * 0.2

    with torch.no_grad():
        torch.manual_seed(1)
        loss_2d = estimator_2d.loss(
            inputs, condition_mask, None, times=times, control_variate=control_variate
        )
        torch.manual_seed(1)
        loss_3d = estimator_3d.loss(
            inputs.unsqueeze(-1),
            condition_mask,
            None,
            times=times,
            control_variate=control_variate,
        )

    assert torch.allclose(loss_2d, loss_3d, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("sde_type", ["ve", "vp", "subvp", "flow"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("net", ["simformer"])
def test_masked_vector_field_estimator_on_device(sde_type, device, net):
    """Test whether MaskedScoreEstimator can be moved to the device."""

    if sde_type == "flow":
        score_estimator = build_masked_flow_matching_estimator(
            torch.randn(100, 5, 1),
            torch.randn(100, 5, 1),
            net=net,
        )
    else:
        score_estimator = build_masked_score_matching_estimator(
            torch.randn(100, 5, 1),
            torch.randn(100, 5, 1),
            sde_type=sde_type,
            net=net,
        )
    score_estimator.to(device)

    # Test forward
    inputs = torch.randn(100, 5, 1, device=device)
    condition_masks = torch.ones(100, 5, device=device)
    edge_masks = torch.ones(100, 5, 5, device=device)
    time = torch.randn(1, device=device)
    out = score_estimator(inputs, time, condition_masks, edge_masks)

    assert str(out.device).split(":")[0] == device, "Output device mismatch."

    # Test loss
    loss = score_estimator.loss(inputs, condition_masks, edge_masks)
    assert str(loss.device).split(":")[0] == device, "Loss device mismatch."


@pytest.mark.parametrize("sde_type", ["ve", "vp", "subvp", "flow"])
@pytest.mark.parametrize("input_sample_dim", (1, 2, 3))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,), (3, 5), (3, 1)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("net", ["simformer"])
def test_masked_vector_field_estimator_forward_shapes(
    sde_type,
    input_sample_dim,
    input_event_shape,
    batch_dim,
    net,
):
    """Test whether `forward` of MaskedScoreEstimator follows the shape convention."""
    (
        score_estimator,
        inputs,
        condition_masks,
        edge_masks,
    ) = _build_masked_vector_field_estimator_and_tensors(
        sde_type,
        input_event_shape,
        batch_dim,
        input_sample_dim,
        net=net,
    )
    # Batched times
    times = torch.rand((batch_dim,))
    outputs = score_estimator(
        inputs[0], time=times, condition_mask=condition_masks, edge_mask=edge_masks
    )
    assert outputs.shape == (
        batch_dim,
        *input_event_shape,
    ), "Output shape mismatch."

    # Single time
    time = torch.rand(())
    outputs = score_estimator(
        inputs[0], time=time, condition_mask=condition_masks, edge_mask=edge_masks
    )
    assert outputs.shape == (
        batch_dim,
        *input_event_shape,
    ), "Output shape mismatch."


def _build_masked_vector_field_estimator_and_tensors(
    sde_type: str,
    input_event_shape: Tuple[int, int],
    batch_dim: int,
    input_sample_dim: int = 1,
    **kwargs,
):
    """
    Helper function for all tests that deal with shapes of masked score estimators.
    """

    num_nodes = input_event_shape[0]

    building_inputs = torch.randn((batch_dim, *input_event_shape))

    if sde_type == "flow":
        score_estimator = build_masked_flow_matching_estimator(
            building_inputs,
            building_inputs,  # not used
            **kwargs,
        )
    else:
        score_estimator = build_masked_score_matching_estimator(
            building_inputs,
            building_inputs,  # not used
            sde_type=sde_type,
            **kwargs,
        )

    inputs = building_inputs[:batch_dim]
    condition_masks = torch.ones(batch_dim, num_nodes)
    condition_masks[:, 1:] = 0  # Index 0 is latent
    edge_masks = torch.ones(batch_dim, num_nodes, num_nodes)

    inputs = inputs.unsqueeze(0)
    inputs = inputs.expand(
        [
            input_sample_dim,
        ]
        + [-1] * (1 + len(input_event_shape))
    )
    return score_estimator, inputs, condition_masks, edge_masks


# *** ======== Unmasked Estimator ======== *** #


@pytest.mark.parametrize("sde_type", ["ve", "vp", "subvp", "flow"])
@pytest.mark.parametrize("input_sample_dim", (1, 2, 3))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,), (3, 5), (3, 1)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("net", ["simformer"])
def test_unmasked_wrapper_vector_field_estimator_loss_shapes(
    sde_type,
    input_sample_dim,
    input_event_shape,
    batch_dim,
    net,
):
    """Test whether `loss` of MaskedConditionalVectorFieldEstimatorWrapper
    follows the shape convention."""
    (
        score_estimator,
        inputs,
        condition,
    ) = _build_unmasked_vector_field_estimator_and_tensors(
        sde_type,
        input_event_shape,
        batch_dim,
        input_sample_dim,
        net=net,
    )

    with pytest.raises(NotImplementedError):
        score_estimator.loss(inputs[0], condition)


@pytest.mark.gpu
@pytest.mark.parametrize("sde_type", ["ve", "vp", "subvp", "flow"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("net", ["simformer"])
def test_unmasked_wrapper_vector_field_estimator_on_device(sde_type, device, net):
    """Test whether MaskedConditionalVectorFieldEstimatorWrapper
    can be moved to the device."""
    # Create condition and edge masks
    condition_mask = torch.ones(5, device=device)
    condition_mask[0] = 0  # Index 0 is latent
    edge_mask = torch.ones(5, 5, device=device)

    building_inputs = torch.randn(100, 5)

    if sde_type == "flow":
        score_estimator = build_masked_flow_matching_estimator(
            building_inputs,
            building_inputs,  # not used
            net=net,
        )
    else:
        score_estimator = build_masked_score_matching_estimator(
            building_inputs,
            building_inputs,  # not used
            sde_type=sde_type,
            net=net,
        )

    score_estimator = score_estimator.to(
        device
    ).build_conditional_vector_field_estimator(condition_mask, edge_mask)

    inputs = torch.randn(100, 1, device=device)
    condition = torch.randn(100, 4, device=device)
    time = torch.randn(1, device=device)
    out = score_estimator(inputs, condition, time)

    assert str(out.device).split(":")[0] == device, "Output device mismatch."


@pytest.mark.gpu
@pytest.mark.parametrize("sde_type", ["ve", "flow"])
def test_unmasked_wrapper_moves_its_index_buffers(sde_type):
    """Moving the wrapper after it was built must also move the latent/observed
    indices, otherwise its forward pass mixes devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    building_inputs = torch.randn(100, 5)
    builder = (
        build_masked_flow_matching_estimator
        if sde_type == "flow"
        else build_masked_score_matching_estimator
    )
    kwargs = {} if sde_type == "flow" else {"sde_type": sde_type}
    condition_mask = torch.tensor([0, 1, 1, 0, 1], dtype=torch.bool)
    wrapper = builder(
        building_inputs, building_inputs, **kwargs
    ).build_conditional_vector_field_estimator(condition_mask, None)

    wrapper = wrapper.to("cuda")
    for name in ("_latent_idx", "_observed_idx", "_assemble_perm"):
        assert getattr(wrapper, name).is_cuda, f"{name} stayed on the CPU."
    out = wrapper(
        torch.randn(7, 2, device="cuda"),
        torch.randn(7, 3, device="cuda"),
        torch.rand(1, device="cuda"),
    )
    assert out.shape == (7, 2) and out.is_cuda


@pytest.mark.parametrize("input_event_shape", ((5,), (5, 3)))
def test_unmasked_wrapper_assembles_inputs_in_node_order(input_event_shape):
    """The out-of-place assembly must place latent and observed entries exactly
    where the condition mask says, and must work under `torch.func.vmap`."""
    torch.manual_seed(0)
    estimator = build_masked_score_matching_estimator(
        torch.randn(50, *input_event_shape), torch.randn(50, *input_event_shape)
    )
    condition_mask = torch.tensor([0, 1, 1, 0, 1], dtype=torch.bool)
    wrapper = estimator.build_conditional_vector_field_estimator(condition_mask, None)
    feat = input_event_shape[1:]
    num_features = feat[0] if feat else 1
    latent = torch.randn(4, 2 * num_features)
    observed = torch.randn(4, 3 * num_features)

    full, batch_shape = wrapper._assemble_full_inputs(latent, observed)

    expected = torch.zeros(4, 5, *feat)
    expected[:, ~condition_mask] = latent.reshape(4, 2, *feat)
    expected[:, condition_mask] = observed.reshape(4, 3, *feat)
    assert batch_shape == (4,)
    assert torch.equal(full, expected)

    # A single latent row is broadcast against several conditions, as the unmasked
    # estimators do (this is what `jac_gauss` relies on).
    full_b, batch_shape_b = wrapper._assemble_full_inputs(latent[0], observed)
    assert batch_shape_b == (4,)
    assert torch.equal(
        full_b[:, ~condition_mask], latent[0].reshape(1, 2, *feat).expand(4, 2, *feat)
    )
    assert torch.equal(full_b[:, condition_mask], expected[:, condition_mask])

    # In-place index writes into a fresh tensor break under vmap; the out-of-place
    # assembly must not.
    batched = torch.func.vmap(
        lambda lat, obs: wrapper._assemble_full_inputs(lat, obs)[0], in_dims=(0, None)
    )(latent, observed)
    assert batched.shape == (4, 4, 5, *feat)
    for i in range(4):
        assert torch.equal(
            batched[i], wrapper._assemble_full_inputs(latent[i], observed)[0]
        )


@pytest.mark.parametrize("sde_type", ["ve", "vp", "subvp", "flow"])
@pytest.mark.parametrize("input_sample_dim", (1, 2, 3))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,), (3, 5), (3, 1)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("net", ["simformer"])
def test_unmasked_wrapper_vector_field_estimator_forward_shapes(
    sde_type,
    input_sample_dim,
    input_event_shape,
    batch_dim,
    net,
):
    """Test whether `forward` of MaskedConditionalVectorFieldEstimatorWrapperù
    follow the shape convention."""
    (
        score_estimator,
        inputs,
        conditions,
    ) = _build_unmasked_vector_field_estimator_and_tensors(
        sde_type,
        input_event_shape,
        batch_dim,
        input_sample_dim,
        net=net,
    )
    # Batched times
    times = torch.rand((batch_dim,))
    outputs = score_estimator(inputs[0], condition=conditions, time=times)

    assert outputs.shape == inputs[0].shape, "Output shape mismatch."

    # Single time
    time = torch.rand(())
    outputs = score_estimator(inputs[0], condition=conditions, time=time)

    assert outputs.shape == inputs[0].shape, "Output shape mismatch."


def _build_unmasked_vector_field_estimator_and_tensors(
    sde_type: str,
    input_event_shape: Tuple[int, int],
    batch_dim: int,
    input_sample_dim: int = 1,
    **kwargs,
):
    """
    Helper function for all tests that deal with shapes of
    unmasked wrapper score estimators.
    """

    (
        score_estimator,
        inputs,
        condition_masks,
        edge_masks,
    ) = _build_masked_vector_field_estimator_and_tensors(
        sde_type,
        input_event_shape,
        batch_dim,
        input_sample_dim,
        **kwargs,
    )

    # # Use the first condition and edge mask for all batches
    condition_masks = condition_masks[0].clone().detach()
    edge_masks = edge_masks[0].clone().detach()

    # Build unmasked score estimator (wrapper)
    score_estimator = score_estimator.build_conditional_vector_field_estimator(
        condition_masks,
        edge_masks,
    )

    # Disassemble inputs
    latent_idx = (condition_masks == 0).squeeze()
    observed_idx = (condition_masks == 1).squeeze()

    # Handle inputs with different number of dimensions
    if len(input_event_shape) == 1:
        untangled_inputs = inputs[:, :, latent_idx]  # (S, B, num_latent)
        untangled_condition = inputs[0, :, observed_idx]  # (B, num_observed)
    else:
        untangled_inputs = inputs[:, :, latent_idx, :]  # (S, B, num_latent, F)
        untangled_condition = inputs[0, :, observed_idx, :]  # (B, num_observed, F)

    return (
        score_estimator,
        untangled_inputs.reshape(input_sample_dim, batch_dim, -1),
        untangled_condition.reshape(batch_dim, -1),
    )


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_train_schedule(config_cls):
    """Test on shapes and bounds for train and solve schedules
    of vector field estimators (flow or score)
    """
    t_min = torch.tensor([0.0])
    t_max = torch.tensor([1.0])

    estimator = config_cls().build(torch.randn(100, 1), torch.randn(100, 1))

    if config_cls is not FlowMatchingConfig:
        # Schedule with default bounds
        train_schedule_default = estimator.train_schedule(300)
        assert train_schedule_default.shape == torch.Size((300,))
        assert train_schedule_default.max() <= estimator.t_max
        assert train_schedule_default.min() >= estimator.t_min

        # Schedule with given bounds
        train_schedule = estimator.train_schedule(300, t_min, t_max)
        assert train_schedule.shape == torch.Size((300,))
        assert train_schedule.max() <= t_max.item()
        assert train_schedule.min() >= t_min.item()

    # Solve schedule with default bounds
    solve_schedule_default = estimator.solve_schedule(
        300, t_max=estimator.t_max, t_min=estimator.t_min
    )
    assert torch.allclose(solve_schedule_default[0], torch.tensor([estimator.t_max]))
    assert torch.allclose(solve_schedule_default[-1], torch.tensor([estimator.t_min]))
    assert solve_schedule_default.shape == torch.Size((300,))
    assert torch.all(solve_schedule_default[:-1] - solve_schedule_default[1:] >= 0)

    # Solve schedule with given bounds
    solve_schedule = estimator.solve_schedule(
        300, t_max=t_max.item(), t_min=t_min.item()
    )
    assert torch.allclose(solve_schedule[0], t_max)
    assert torch.allclose(solve_schedule[-1], t_min)
    assert solve_schedule_default.shape == torch.Size((300,))
    assert torch.all(solve_schedule[:-1] - solve_schedule[1:] >= 0)


@pytest.mark.parametrize(
    "train_schedule,solve_schedule",
    [
        ("uniform", "uniform"),
        ("lognormal", "uniform"),
        ("uniform", "power_law"),
        ("lognormal", "power_law"),
    ],
)
def test_ve_edm_schedules(train_schedule, solve_schedule):
    """Test EDM-style schedules for VE estimator (Karras et al. 2022)."""
    estimator = VEScoreConfig(
        train_schedule=train_schedule,
        solve_schedule=solve_schedule,
    ).build(torch.randn(100, 1), torch.randn(100, 1))

    # Test train schedule returns valid times without NaN.
    times_train = estimator.train_schedule(500)
    assert times_train.shape == (500,)
    assert torch.all(times_train >= estimator.t_min), "Train times below t_min"
    assert torch.all(times_train <= estimator.t_max), "Train times above t_max"
    assert not torch.any(torch.isnan(times_train)), "NaN in train schedule"

    # Test solve schedule returns monotonically decreasing times without NaN.
    times_solve = estimator.solve_schedule(100)
    assert times_solve.shape == (100,)
    assert torch.allclose(times_solve[0], torch.tensor(estimator.t_max)), (
        "First solve time != t_max"
    )
    assert torch.allclose(times_solve[-1], torch.tensor(estimator.t_min)), (
        "Last solve time != t_min"
    )
    assert torch.all(times_solve[:-1] >= times_solve[1:]), (
        "Solve schedule not monotonically decreasing"
    )
    assert not torch.any(torch.isnan(times_solve)), "NaN in solve schedule"


def test_ve_lognormal_no_nan_with_extreme_params():
    """Test that lognormal schedule doesn't produce NaN even with extreme params."""
    # Use parameters that could cause extreme sigma values.
    estimator = VEScoreConfig(
        train_schedule="lognormal",
        lognormal_mean=-3.0,  # Very low mean -> small sigmas
        lognormal_std=2.0,  # High variance -> some extreme samples
    ).build(torch.randn(100, 1), torch.randn(100, 1))

    # Generate many samples to test edge cases.
    times = estimator.train_schedule(10000)
    assert not torch.any(torch.isnan(times)), (
        "NaN produced with extreme lognormal params"
    )
    assert not torch.any(torch.isinf(times)), (
        "Inf produced with extreme lognormal params"
    )
    assert torch.all(times >= estimator.t_min), "Times below t_min"
    assert torch.all(times <= estimator.t_max), "Times above t_max"


@pytest.mark.parametrize("iid_method", ["fnpe", "gauss", "auto_gauss", "jac_gauss"])
@pytest.mark.parametrize("num_features", [None, 2])
def test_simformer_iid_sampling_runs(iid_method, num_features):
    """Posterior sampling with several iid observations must work for every iid
    method, including `jac_gauss`, which vmaps through the fixed-mask wrapper."""
    from sbi.inference import Simformer
    from sbi.inference.posteriors.posterior_parameters import (
        VectorFieldPosteriorParameters,
    )

    torch.manual_seed(0)
    num_theta, num_x = 2, 3
    shape = () if num_features is None else (num_features,)
    theta = torch.randn(200, num_theta, *shape)
    x = theta.sum(1, keepdim=True).expand(-1, num_x, *shape) + 0.1 * torch.randn(
        200, num_x, *shape
    )
    inference = Simformer(
        posterior_latent_idx=list(range(num_theta)),
        posterior_observed_idx=list(range(num_theta, num_theta + num_x)),
        hidden_features=16,
        num_layers=1,
        show_progress_bars=False,
    )
    inference.append_simulations(torch.cat([theta, x], dim=1))
    inference.train(max_num_epochs=2)
    num_latent_entries = num_theta * (num_features or 1)
    prior = torch.distributions.Independent(
        torch.distributions.Normal(
            torch.zeros(num_latent_entries), torch.ones(num_latent_entries)
        ),
        1,
    )
    posterior = inference.build_posterior(
        prior=prior,  # The iid methods need the prior.
        sample_with="sde",
        posterior_parameters=VectorFieldPosteriorParameters(iid_method=iid_method),
    )

    x_o = x[:3].reshape(3, -1)  # three iid observations
    samples = posterior.sample((10,), x=x_o, steps=5, show_progress_bars=False)
    assert samples.shape == (10, num_theta * (num_features or 1))
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize(
    "sde_type, estimator_kwargs, expected",
    [
        (
            "ve",
            dict(sigma_max=50.0, t_min=0.1, weight_fn="identity"),
            dict(sigma_max=50.0, t_min=0.1),
        ),
        ("vp", dict(beta_max=5.0, t_max=0.9), dict(beta_max=5.0, t_max=0.9)),
        ("subvp", dict(beta_min=0.2), dict(beta_min=0.2)),
        ("flow", dict(noise_scale=1e-2), dict(noise_scale=1e-2)),
    ],
)
def test_simformer_factory_passes_estimator_options(
    sde_type, estimator_kwargs, expected
):
    """Estimator options given to the Simformer factories must reach the estimator
    (they used to be silently dropped), and network options the network."""
    from sbi.neural_nets.factory import simformer_flow_nn, simformer_score_nn

    if sde_type == "flow":
        build_fn = simformer_flow_nn(hidden_features=24, **estimator_kwargs)
    else:
        build_fn = simformer_score_nn(
            sde_type=sde_type, hidden_features=24, **estimator_kwargs
        )
    estimator = build_fn(torch.randn(50, 4))

    for name, value in expected.items():
        assert getattr(estimator, name) == pytest.approx(value), name
    if "weight_fn" in estimator_kwargs:
        weights = torch.as_tensor(estimator.weight_fn(torch.rand(5)), dtype=torch.float)
        assert torch.allclose(weights, torch.ones(()))
    # The network option reached the network: a narrower net has fewer parameters.
    default_net = (simformer_flow_nn() if sde_type == "flow" else simformer_score_nn())(
        torch.randn(50, 4)
    ).net
    num_params = sum(p.numel() for p in estimator.net.parameters())
    assert num_params < sum(p.numel() for p in default_net.parameters())


@pytest.mark.parametrize("sde_type", ["ve", "flow"])
def test_masked_builder_rejects_unknown_and_unused_options(sde_type):
    """Typos and options a custom network cannot use must raise, not be dropped."""
    builder = (
        build_masked_flow_matching_estimator
        if sde_type == "flow"
        else build_masked_score_matching_estimator
    )
    data = torch.randn(50, 4)
    with pytest.raises(ValueError, match="sigma_maxx"):
        builder(data, data, sigma_maxx=3.0)

    custom_net = builder(data, data).net
    with pytest.raises(ValueError, match="hidden_features"):
        builder(data, data, net=custom_net, hidden_features=8)


def test_finite_z_score_stats_ignore_nans_per_entry():
    """Z-scoring statistics use the finite values of each entry, so they also work
    when every row has a missing value, and match `z_standardization` without
    NaNs."""
    from sbi.neural_nets.net_builders.vector_field_nets import _finite_z_score_stats
    from sbi.utils.sbiutils import z_standardization

    torch.manual_seed(0)
    data = torch.randn(400, 3) * torch.tensor([1.0, 10.0, 0.1]) + 5.0
    mean, std = _finite_z_score_stats(data, structured=False)
    ref_mean, ref_std = z_standardization(data, structured_dims=False)
    assert torch.allclose(mean, ref_mean, atol=1e-5)
    assert torch.allclose(std, ref_std, atol=1e-5)

    with_nans = data.clone()
    with_nans[torch.arange(400), torch.arange(400) % 3] = float("nan")  # every row
    mean_nan, std_nan = _finite_z_score_stats(with_nans, structured=False)
    assert torch.isfinite(mean_nan).all() and torch.isfinite(std_nan).all()
    assert torch.allclose(mean_nan, mean, rtol=0.1)
    assert torch.allclose(std_nan, std, rtol=0.15)


def test_flow_simformer_z_scoring_samples_in_original_space():
    """With z-scoring, the flow Simformer is trained in standardized space; the
    posterior must still return samples in the original space."""
    from sbi.inference import FlowMatchingSimformer

    torch.manual_seed(0)
    num_sims = 500
    theta = 100.0 + 20.0 * torch.randn(num_sims, 2)
    x = 1000.0 + 5.0 * (theta - 100.0) + torch.randn(num_sims, 2)
    inference = FlowMatchingSimformer(
        posterior_latent_idx=[0, 1],
        posterior_observed_idx=[2, 3],
        z_score_inputs="independent",
        hidden_features=32,
        num_layers=2,
        show_progress_bars=False,
    )
    inference.append_simulations(torch.cat([theta, x], dim=1))
    estimator = inference.train(max_num_epochs=30)
    assert estimator.z_score_enabled

    posterior = inference.build_posterior(sample_with="ode")
    samples = posterior.sample((200,), x=x[:1], show_progress_bars=False)
    assert torch.isfinite(samples).all()
    # theta is about (x - 1000) / 5 + 100; samples must be on that scale.
    expected = (x[0] - 1000.0) / 5.0 + 100.0
    assert (samples.mean(0) - expected).abs().max() < 20.0

    # Standardized space does not support iid observations yet; it must say so.
    with pytest.raises(NotImplementedError):
        posterior.sample((5,), x=x[:2], show_progress_bars=False)


def test_finite_z_score_stats_fall_back_for_too_few_values():
    """Entries with fewer than two finite values are left unscaled, with a
    warning, so that building an estimator from tiny batches still works."""
    from sbi.neural_nets.net_builders.vector_field_nets import _finite_z_score_stats

    data = torch.tensor([[1.0, 2.0], [3.0, float("nan")]])
    with pytest.warns(UserWarning, match="fewer than two finite values"):
        mean, std = _finite_z_score_stats(data, structured=False)
    assert torch.allclose(mean, torch.tensor([2.0, 2.0]))
    assert torch.allclose(std, torch.tensor([2.0**0.5, 1.0]))
