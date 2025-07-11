import inspect

import pytest

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
            {"potential_fn", "proposal", "device", "theta_transform"},
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
    dataclass_signature = inspect.signature(parameter_dataclass)
    class_signature = inspect.signature(init_target_class.__init__)

    skipped_fields_and_parameters.add("self")

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
