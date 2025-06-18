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


@pytest.mark.parametrize(
    ("parameter_dataclass", "posterior_class", "skipped_arguments"),
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
            {"vector_field_estimator", "device", "prior", "sample_with"},
        ),
    ],
)
def test_signature_consistency(parameter_dataclass, posterior_class, skipped_arguments):
    dataclass_signature = inspect.signature(parameter_dataclass)
    class_signature = inspect.signature(posterior_class.__init__)

    skipped_arguments.add("self")

    class_dict = {
        name: param
        for name, param in class_signature.parameters.items()
        if name not in skipped_arguments and param.kind != inspect.Parameter.VAR_KEYWORD
    }

    dataclass_dict = {
        name: param
        for name, param in dataclass_signature.parameters.items()
        if name not in skipped_arguments
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
            f"Default mismatch for '{name}': "
            f"class={class_default}, dataclass={dataclass_default}"
        )
        assert class_annotation == dataclass_annotation, (
            f"Annotation mismatch for '{name}': "
            f"class={class_annotation}, dataclass={dataclass_annotation}"
        )
