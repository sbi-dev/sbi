import pytest

from sbi.utils.get_nn_models import critic_nn, likelihood_nn, posterior_nn


@pytest.mark.parametrize(
    "model", ["linear", "mlp", "resnet"], ids=["linear", "mlp", "resnet"]
)
def test_deprecated_import_critic_nn(model: str):
    with pytest.warns(DeprecationWarning):
        build_fcn = critic_nn(model)
        assert callable(build_fcn)


@pytest.mark.parametrize(
    "model",
    ["mdn", "made", "maf", "maf_rqs", "nsf", "mnle", "zuko_maf"],
    ids=["mdn", "made", "maf", "maf_rqs", "nsf", "mnle", "zuko_maf"],
)
def test_deprecated_import_likelihood_nn(model: str):
    with pytest.warns(DeprecationWarning):
        build_fcn = likelihood_nn(model)
        assert callable(build_fcn)


@pytest.mark.parametrize(
    "model",
    ["mdn", "made", "maf", "maf_rqs", "nsf", "mnle", "zuko_maf"],
    ids=["mdn", "made", "maf", "maf_rqs", "nsf", "mnle", "zuko_maf"],
)
def test_deprecated_import_posterior_nn(model: str):
    with pytest.warns(DeprecationWarning):
        build_fcn = posterior_nn(model)
        assert callable(build_fcn)
