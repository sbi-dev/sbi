# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import inspect
from dataclasses import fields

import pytest
import torch

from sbi.inference import NRE
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.posteriors.importance_posterior import ImportanceSamplingPosterior
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.posteriors.posterior_parameters import (
    DirectPosteriorParameters,
    ImportanceSamplingPosteriorParameters,
    MCMCPosteriorParameters,
    RejectionPosteriorParameters,
    VIPosteriorParameters,
    VectorFieldPosteriorParameters,
)
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.posteriors.vector_field_posterior import VectorFieldPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.inference.potentials.vector_field_potential import VectorFieldBasedPotential
from sbi.utils.torchutils import BoxUniform


@pytest.fixture(scope="session")
def get_inference():
    def simulator(theta):
        return 1.0 + theta + torch.randn(theta.shape, device=theta.device) * 0.1

    num_dim = 3
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
    theta = prior.sample((300,))
    x = simulator(theta)

    inference = NRE(prior=prior)
    inference.append_simulations(theta, x)
    inference.train(max_num_epochs=1)
    return inference


@pytest.mark.parametrize(
    ("parameter_dataclass", "init_target_class", "skipped_fields_and_parameters"),
    [
        (
            DirectPosteriorParameters,
            DirectPosterior,
            {"posterior_estimator", "prior", "device"},
        ),
        (
            ImportanceSamplingPosteriorParameters,
            ImportanceSamplingPosterior,
            {"potential_fn", "proposal", "device"},
        ),
        (
            MCMCPosteriorParameters,
            MCMCPosterior,
            {
                "potential_fn",
                "proposal",
                "device",
                "theta_transform",
                "init_strategy_num_candidates",
            },
        ),
        (
            RejectionPosteriorParameters,
            RejectionPosterior,
            {"potential_fn", "device", "proposal"},
        ),
        (
            VIPosteriorParameters,
            VIPosterior,
            {"potential_fn", "prior", "theta_transform", "device"},
        ),
        (
            VectorFieldPosteriorParameters,
            VectorFieldPosterior,
            {
                "vector_field_estimator",
                "device",
                "prior",
                "sample_with",
                "iid_method",
                "iid_params",
                "neural_ode_backend",
                "neural_ode_kwargs",
            },
        ),
        (
            VectorFieldPosteriorParameters,
            VectorFieldBasedPotential,
            {
                "vector_field_estimator",
                "device",
                "prior",
                "x_o",
                "enable_transform",
                "max_sampling_batch_size",
            },
        ),
    ],
)
def test_signature_consistency(
    parameter_dataclass, init_target_class, skipped_fields_and_parameters
):
    """
    Test that the constructor (__init__) signature of a target class matches the
    signature of a corresponding parameter dataclass.

    This function compares the argument names, default values, and type annotations
    between the dataclass and the target class __init__ method, ignoring specified
    parameters passed in `skipped_fields_and_parameters`.

    Args:
        parameter_dataclass: The dataclass whose signature is used as reference.
        init_target_class: The class whose __init__ method signature is compared.
        skipped_fields_and_parameters (set): A set of parameter names to ignore during
            comparison (e.g., 'self', or fields not relevant for matching).

    Raises:
        AssertionError: If there is any mismatch in parameter names, default values,
            or type annotations between the dataclass and the class constructor.
    """
    dataclass_signature = inspect.signature(parameter_dataclass)
    class_signature = inspect.signature(init_target_class.__init__)

    skipped_fields_and_parameters.add("self")
    skipped_fields_and_parameters.add("x_shape")

    class_dict = {
        name: param
        for name, param in class_signature.parameters.items()
        if name not in skipped_fields_and_parameters
        and param.kind != inspect.Parameter.VAR_KEYWORD
    }

    dataclass_dict = {
        name: param
        for name, param in dataclass_signature.parameters.items()
        if name not in skipped_fields_and_parameters
    }

    # Compare if the dataclass and posterior_class have the same argument names
    assert class_dict.keys() == dataclass_dict.keys(), (
        f"Parameter mismatch:\n"
        f"In class but not dataclass: {class_dict.keys() - dataclass_dict.keys()}\n"
        f"In dataclass but not class: {dataclass_dict.keys() - class_dict.keys()}"
    )

    # Compare if the dataclass and posterior_class
    # have the same annotation and default value
    for name in class_dict:
        class_default, class_annotation = (
            class_dict[name].default,
            class_dict[name].annotation,
        )
        dataclass_default, dataclass_annotation = (
            dataclass_dict[name].default,
            dataclass_dict[name].annotation,
        )
        assert class_default == dataclass_default, (
            f"Default value mismatch for '{name}': "
            f"class={class_default}, dataclass={dataclass_default}"
        )
        assert class_annotation == dataclass_annotation, (
            f"Annotation mismatch for '{name}': "
            f"class={class_annotation}, dataclass={dataclass_annotation}"
        )


@pytest.mark.parametrize(
    "build_posterior_arguments",
    [
        dict(
            mcmc_method="slice_pymc",
            posterior_parameters=MCMCPosteriorParameters(method="hmc_pyro"),
        ),
        dict(
            vi_method="IW",
            posterior_parameters=VIPosteriorParameters(vi_method="fKL"),
        ),
    ],
)
def test_build_posterior_warns_on_conflicting_args(
    build_posterior_arguments, get_inference
):
    """
    Test that build_posterior raises a UserWarning on conflicting parameter
    combinations.
    """
    inference = get_inference

    with pytest.warns(UserWarning, match="ignored in favor of"):
        inference.build_posterior(**build_posterior_arguments)


@pytest.mark.parametrize(
    "build_posterior_arguments",
    [
        pytest.param(
            dict(
                posterior_parameters=MCMCPosteriorParameters(
                    method="slice_np_vectorized"
                ),
            ),
        ),
        pytest.param(
            dict(
                posterior_parameters=VIPosteriorParameters(vi_method="rKL"),
            ),
        ),
    ],
)
def test_build_posterior_works_on_default_args(
    build_posterior_arguments, get_inference
):
    """
    Test that build_posterior doesn't raise on default parameters.
    """

    inference = get_inference
    inference.build_posterior(**build_posterior_arguments)


@pytest.mark.parametrize(
    ("build_posterior_arguments"),
    [
        pytest.param(
            dict(
                mcmc_parameters=dict(num_chains=1),
                posterior_parameters=MCMCPosteriorParameters(),
            ),
            marks=pytest.mark.xfail(
                raises=ValueError,
                reason="Parameter dictionary and posterior_parameter dataclass"
                " shouldn't be passed together.",
            ),
        ),
        pytest.param(
            dict(
                posterior_parameters={},
            ),
            marks=pytest.mark.xfail(
                raises=TypeError,
                reason="Posterior parameters is expected to be type of"
                " PosteriorParameters dataclass",
            ),
        ),
    ],
)
def test_build_posterior_conflicting_params(build_posterior_arguments, get_inference):
    """
    Test whether build_posterior properly raises errors when incorrect
    posterior_parameters values are passed.
    """
    inference = get_inference
    inference.build_posterior(**build_posterior_arguments)


def test_posterior_parameters_with_param_returns_the_same_type():
    """
    Test that calling `with_param()` without arguments returns an instance
    of the same type and with the same values as the original.
    """

    posterior_parameters = MCMCPosteriorParameters()
    new_posterior_parameters = posterior_parameters.with_param()

    for field in posterior_parameters.__dataclass_fields__.values():
        original_value = getattr(posterior_parameters, field.name)
        new_value = getattr(new_posterior_parameters, field.name)

        assert original_value == new_value, f"Mismatch in field '{field.name}'"


def test_posterior_parameters_with_param_updates_value():
    """
    Test that `with_param()` correctly updates specified fields while keeping
    other values unchanged.
    """

    posterior_parameters = MCMCPosteriorParameters(warmup_steps=100)
    new_posterior_parameters = posterior_parameters.with_param(warmup_steps=10)

    assert (
        posterior_parameters.warmup_steps == 100
        and new_posterior_parameters.warmup_steps == 10
    )


@pytest.mark.xfail(
    raises=ValueError, reason="steps field does't exist in MCMCPosteriorParameters"
)
def test_posterior_parameters_fails_for_incorrect_parameter():
    """
    Test that `with_param()` raises a ValueError when an invalid field
    (not defined in the dataclass) is passed.
    """

    posterior_parameters = MCMCPosteriorParameters()
    _ = posterior_parameters.with_param(steps=10)


@pytest.mark.parametrize(
    "param_class",
    [
        DirectPosteriorParameters,
        ImportanceSamplingPosteriorParameters,
        MCMCPosteriorParameters,
        RejectionPosteriorParameters,
        VectorFieldPosteriorParameters,
        VIPosteriorParameters,
    ],
)
def test_valid_field_values(param_class):
    """
    Test that each PosteriorParameters subclass works when passed
    default values for its fields.
    """

    for field in fields(param_class):
        valid_kwarg = {}
        valid_kwarg[field.name] = field.default

        param_class(**valid_kwarg)


@pytest.mark.parametrize(
    ("posterior_parameter_class", "parameter_name", "value", "expected_type"),
    [
        pytest.param(
            RejectionPosteriorParameters,
            "num_iter_to_find_max",
            100.0,
            int,
        ),
        pytest.param(
            RejectionPosteriorParameters,
            "m",
            1,
            float,
        ),
        pytest.param(
            DirectPosteriorParameters,
            "enable_transform",
            0,
            bool,
        ),
    ],
)
def test_valid_primitive_type_conversion(
    posterior_parameter_class, parameter_name, value, expected_type
):
    """
    Test whether primitive types are properly converted to their field annotation types.
    """

    posterior_parameter = posterior_parameter_class(**{parameter_name: value})
    field_value = getattr(posterior_parameter, parameter_name)
    assert isinstance(field_value, expected_type), (
        f"Expected parameter type={expected_type} but got {type(field_value)}",
    )


@pytest.mark.xfail(raises=ValueError, reason="Type conversion failure")
def test_invalid_primitive_conversion_failure():
    """
    Test whether primitive types conversion fails when invalid type is passed.
    """

    _ = RejectionPosteriorParameters(m="a")


@pytest.mark.xfail(
    raises=ValueError, reason="Invalid literal type passed to PosteriorParameters class"
)
def test_invalid_literal_field_values():
    """
    Test PosteriorParameters class construction when invalid literal
    type is passed to a Literal type field.
    """

    MCMCPosteriorParameters(method="invalid")


@pytest.mark.parametrize(
    "params",
    [
        dict(mcmc_method="slice_pymc"),
        dict(vi_method="fKL"),
        dict(vi_parameters={}),
        dict(mcmc_parameters={"thin": 10}),
    ],
)
def test_if_warning_raised_for_deprecated_build_posterior_parameters(
    params, get_inference
):
    """
    Check if the build_posterior method raises a warning for deprecated parameters
    """

    with pytest.warns(FutureWarning, match="The following arguments are deprecated"):
        get_inference.build_posterior(**params)
