import pytest

from sbi.utils.get_nn_models import classifier_nn, likelihood_nn, posterior_nn

models_to_test = [
    "mdn",
    "made",
    "maf",
    "maf_rqs",
    "nsf",
    "mnle",
    "zuko_bpf",
    "zuko_gf",
    "zuko_maf",
    "zuko_naf",
    "zuko_ncsf",
    "zuko_nice",
    "zuko_nsf",
    "zuko_sospf",
    "zuko_unaf",
]


@pytest.mark.parametrize(
    "model", ["linear", "mlp", "resnet"], ids=["linear", "mlp", "resnet"]
)
def test_deprecated_import_classifier_nn(model: str):
    with pytest.warns(DeprecationWarning):
        build_fcn = classifier_nn(model)
        assert callable(build_fcn)


@pytest.mark.parametrize("model", models_to_test, ids=models_to_test)
def test_deprecated_import_likelihood_nn(model: str):
    with pytest.warns(DeprecationWarning):
        build_fcn = likelihood_nn(model)
        assert callable(build_fcn)


@pytest.mark.parametrize("model", models_to_test, ids=models_to_test)
def test_deprecated_import_posterior_nn(model: str):
    with pytest.warns(DeprecationWarning):
        build_fcn = posterior_nn(model)
        assert callable(build_fcn)
