# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings

import pytest
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import MNLE, MNPE, NLE_A, NPE_A, NPE_C, NPE_PFN
from sbi.neural_nets import likelihood_nn, posterior_nn
from sbi.neural_nets.estimators import MixedDensityEstimator
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.net_builders.estimator_configs import (
    _MIXED_CONTINUOUS_CONFIGS,
    MAFConfig,
    MDNConfig,
    MixedConfig,
    NSFConfig,
    ResNetClassifierConfig,
    TabPFNConfig,
    VectorFieldEstimatorBuilder,
    ZukoNSFConfig,
)
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import check_estimator_arg

_TRAINERS = [(NPE_C, posterior_nn, "theta"), (NLE_A, likelihood_nn, "x")]


@pytest.mark.parametrize(
    "trainer_cls,factory_fn",
    [(t, f) for t, f, _ in _TRAINERS],
    ids=["npe", "nle"],
)
def test_no_warning_for_valid_inputs(trainer_cls, factory_fn):
    """None default, config, and callable should not emit FutureWarning."""
    prior = MultivariateNormal(zeros(2), eye(2))

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        trainer_cls(prior, show_progress_bars=False)
        trainer_cls(prior, density_estimator=MAFConfig(), show_progress_bars=False)
        trainer_cls(
            prior,
            density_estimator=factory_fn(model="maf"),
            show_progress_bars=False,
        )


@pytest.mark.parametrize(
    "trainer_cls",
    [t for t, _, _ in _TRAINERS],
    ids=["npe", "nle"],
)
def test_string_emits_deprecation_warning(trainer_cls):
    """Passing a string to density_estimator should emit FutureWarning."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.warns(FutureWarning, match="from sbi.neural_nets import"):
        trainer_cls(prior, density_estimator="maf", show_progress_bars=False)


@pytest.mark.parametrize(
    "trainer_cls,input_var",
    [(t, v) for t, _, v in _TRAINERS],
    ids=["npe", "nle"],
)
def test_train_with_config(trainer_cls, input_var):
    """Train with a per-model config, verify loss and posterior sampling."""
    num_dim_theta, num_dim_x = 2, 5
    prior = MultivariateNormal(zeros(num_dim_theta), eye(num_dim_theta))
    config = MAFConfig(hidden_features=16, num_transforms=2)
    inference = trainer_cls(prior, density_estimator=config, show_progress_bars=False)

    theta = prior.sample((100,))
    x = torch.randn(100, num_dim_x)
    density_estimator = inference.append_simulations(theta, x).train(
        max_num_epochs=1, training_batch_size=50
    )

    assert density_estimator.input_shape == torch.Size([
        num_dim_theta if input_var == "theta" else num_dim_x
    ])
    assert density_estimator.condition_shape == torch.Size([
        num_dim_x if input_var == "theta" else num_dim_theta
    ])

    # Verify finite loss on a fresh batch with correct role order.
    fresh_theta = prior.sample((10,))
    fresh_x = torch.randn(10, num_dim_x)
    if input_var == "theta":
        # NPE: loss(input=θ, condition=x)
        loss = density_estimator.loss(fresh_theta, condition=fresh_x)
    else:
        # NLE: loss(input=x, condition=θ)
        loss = density_estimator.loss(fresh_x, condition=fresh_theta)
    assert loss.shape == (10,)
    assert torch.isfinite(loss).all()

    # Posterior should be constructable and produce correct-shaped samples.
    posterior = inference.build_posterior()
    x_o = zeros(1, num_dim_x)
    samples = posterior.sample((10,), x=x_o)
    assert samples.shape == (10, num_dim_theta)


@pytest.mark.parametrize("trainer_cls", [MNLE, MNPE], ids=["mnle", "mnpe"])
def test_mixed_no_warning_for_valid_inputs(trainer_cls):
    """None default and config should not emit FutureWarning."""
    prior = MultivariateNormal(zeros(2), eye(2))

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        trainer_cls(prior, show_progress_bars=False)
        trainer_cls(prior, density_estimator=MixedConfig(), show_progress_bars=False)


@pytest.mark.parametrize(
    "trainer_cls,string",
    [(MNLE, "mnle"), (MNPE, "mnpe")],
    ids=["mnle", "mnpe"],
)
def test_mixed_string_emits_deprecation_warning(trainer_cls, string):
    """Passing 'mnle'/'mnpe' string should emit FutureWarning."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.warns(FutureWarning, match="deprecated"):
        trainer_cls(prior, density_estimator=string, show_progress_bars=False)


@pytest.mark.parametrize(
    "trainer_cls,wrong_string",
    [(MNLE, "maf"), (MNPE, "nsf")],
    ids=["mnle", "mnpe"],
)
def test_mixed_wrong_string_raises(trainer_cls, wrong_string):
    """Passing a non-mnle/mnpe string should raise ValueError."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.raises(ValueError, match="supports only"):
        trainer_cls(prior, density_estimator=wrong_string, show_progress_bars=False)


@pytest.mark.parametrize(
    "continuous",
    [ResNetClassifierConfig(), MixedConfig()],
    ids=["classifier", "mixed"],
)
def test_mixed_rejects_a_continuous_model_it_cannot_build(continuous):
    """The nested config replaces the old `continuous_model` Literal.

    That Literal quietly restricted which models MNLE and MNPE accept, so the
    restriction has to be re-stated as a type check on the nested config.
    """
    with pytest.raises(TypeError, match="continuous component"):
        MixedConfig(continuous=continuous)


def test_mixed_continuous_configs_match_the_build_functions():
    """Guard against drift between the allowed set and `mixed_nets`."""
    from sbi.neural_nets.net_builders.estimator_configs import _DENSITY_CONFIGS
    from sbi.neural_nets.net_builders.mixed_nets import model_builders

    assert (
        frozenset(_DENSITY_CONFIGS[name] for name in model_builders)
        == _MIXED_CONTINUOUS_CONFIGS
    )


def test_mixed_requires_explicit_auxiliary_widths_for_per_transform_widths():
    """Fallback widths require the continuous config to hold one integer."""
    with pytest.raises(ValueError, match="hidden_features"):
        MixedConfig(continuous=ZukoNSFConfig(hidden_features=[16, 16]))


def test_mixed_accepts_per_transform_widths_with_explicit_auxiliary_widths():
    """The categorical and combined nets can be sized independently."""
    config = MixedConfig(
        continuous=ZukoNSFConfig(hidden_features=[16, 16]),
        discrete_hidden_features=12,
        combined_embedding_features=14,
    )
    theta = torch.cat(
        [torch.randn(100, 2), torch.randint(0, 3, (100, 1)).float()], dim=-1
    )
    x = torch.randn(100, 4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator = config.build(theta, x)

    assert isinstance(estimator, MixedDensityEstimator)


@pytest.mark.parametrize(
    "continuous",
    [
        NSFConfig(z_score_condition="structured"),
        NSFConfig(embedding_net=torch.nn.Linear(4, 4)),
    ],
    ids=["z-score", "embedding"],
)
def test_mixed_rejects_nested_condition_settings_it_replaces(continuous):
    """Nested condition settings must not be accepted and then discarded."""
    with pytest.raises(ValueError, match="replaced"):
        MixedConfig(continuous=continuous)


def test_mixed_rejects_unroutable_extra_kwargs():
    with pytest.raises(ValueError, match="no downstream pass-through"):
        MixedConfig(extra_kwargs={"typo": 1})


def test_mixed_dropout_reaches_the_categorical_net():
    theta = torch.cat(
        [torch.randn(100, 2), torch.randint(0, 3, (100, 1)).float()], dim=-1
    )
    x = torch.randn(100, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator = MixedConfig(dropout_probability=0.4).build(theta, x)

    dropout = [
        m.p for m in estimator.discrete_net.modules() if isinstance(m, torch.nn.Dropout)
    ]
    assert dropout and set(dropout) == {0.4}


def test_mixed_default_keeps_the_previous_continuous_net():
    """Nesting must not quietly change the network MNLE and MNPE have shipped.

    The mixed build function overrode the spline tail bound with its own value,
    so the default continuous config has to carry that rather than `NSFConfig`'s.
    """
    import inspect

    from sbi.neural_nets.net_builders.mixed_nets import (
        _build_mixed_density_estimator,
    )

    params = inspect.signature(_build_mixed_density_estimator).parameters
    continuous = MixedConfig().continuous

    assert isinstance(continuous, NSFConfig)
    assert continuous.z_score_input == params["z_score_x"].default
    assert continuous.tail_bound == params["tail_bound"].default
    assert continuous.hidden_features == params["hidden_features"].default
    assert continuous.num_transforms == params["num_transforms"].default
    assert continuous.num_bins == params["num_bins"].default
    assert MixedConfig().dropout_probability == params["dropout_probability"].default


@pytest.mark.parametrize(
    "trainer_cls,make_data,input_var",
    [
        (
            MNLE,
            # MNLE: theta=continuous params, x=mixed observations
            lambda n: (
                BoxUniform(low=zeros(2), high=torch.ones(2)).sample((n,)),
                torch.cat(
                    [torch.randn(n, 3), torch.randint(0, 3, (n, 2)).float()], dim=-1
                ),
            ),
            "x",
        ),
        (
            MNPE,
            # MNPE: theta=mixed params, x=continuous observations
            lambda n: (
                torch.cat(
                    [torch.randn(n, 2), torch.randint(0, 3, (n, 1)).float()], dim=-1
                ),
                torch.randn(n, 4),
            ),
            "theta",
        ),
    ],
    ids=["mnle", "mnpe"],
)
def test_mixed_train_with_config(trainer_cls, make_data, input_var):
    """Train MNLE/MNPE with a nested continuous config end-to-end."""
    config = MixedConfig(
        continuous=NSFConfig(hidden_features=16, num_transforms=2, tail_bound=10.0)
    )
    trainer = trainer_cls(density_estimator=config, show_progress_bars=False)

    theta, x = make_data(200)
    estimator = trainer.append_simulations(theta, x).train(
        max_num_epochs=2, training_batch_size=100
    )
    assert isinstance(estimator, MixedDensityEstimator)

    # verify the estimator actually evaluates with finite loss
    theta_fresh, x_fresh = make_data(10)
    if input_var == "theta":
        loss = estimator.loss(theta_fresh, condition=x_fresh)
    else:
        loss = estimator.loss(x_fresh, condition=theta_fresh)
    assert loss.shape == (10,)
    assert torch.isfinite(loss).all()


def test_mixed_continuous_settings_reach_the_continuous_net():
    """A setting on the nested config must configure the continuous sub-net."""
    theta = torch.cat(
        [torch.randn(200, 2), torch.randint(0, 3, (200, 1)).float()], dim=-1
    )
    x = torch.randn(200, 4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        narrow = MixedConfig(continuous=NSFConfig(hidden_features=16)).build(
            batch_input=theta, batch_condition=x
        )
        wide = MixedConfig(continuous=NSFConfig(hidden_features=64)).build(
            batch_input=theta, batch_condition=x
        )

    assert sum(p.numel() for p in narrow.continuous_net.parameters()) < sum(
        p.numel() for p in wide.continuous_net.parameters()
    )
    assert sum(p.numel() for p in narrow.discrete_net.parameters()) < sum(
        p.numel() for p in wide.discrete_net.parameters()
    )


@pytest.mark.parametrize(
    "estimator",
    (
        MAFConfig(),
        MixedConfig(),
        "maf",
        posterior_nn(model="maf"),
    ),
    ids=["density_config", "mixed_config", "string", "callable"],
)
def test_check_estimator_arg_accepts_valid_inputs(estimator):
    """check_estimator_arg should accept configs, strings, and callables."""
    check_estimator_arg(estimator)


def test_check_estimator_arg_rejects_module():
    """check_estimator_arg should reject raw nn.Module instances."""
    with pytest.raises(TypeError):
        check_estimator_arg(torch.nn.Linear(3, 3))


@pytest.mark.parametrize("config_cls", [MAFConfig, VectorFieldEstimatorBuilder])
def test_check_estimator_arg_rejects_config_class(config_cls):
    """A forgotten pair of parentheses must fail before training."""
    with pytest.raises(TypeError, match="not an instance"):
        check_estimator_arg(config_cls)


@pytest.mark.parametrize(
    "trainer_cls", [t for t, _, _ in _TRAINERS], ids=["npe", "nle"]
)
@pytest.mark.parametrize(
    "config", [ResNetClassifierConfig(), MixedConfig()], ids=["classifier", "mixed"]
)
def test_wrong_config_family_raises(trainer_cls, config):
    """Configs of the wrong family should raise TypeError early."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.raises(TypeError, match="DensityConfigBase"):
        trainer_cls(prior, density_estimator=config)


@pytest.mark.parametrize(
    "trainer_cls",
    [NPE_C, NLE_A, MNPE, MNLE, NPE_PFN],
    ids=["npe", "nle", "mnpe", "mnle", "npe_pfn"],
)
def test_trainer_rejects_legacy_config_of_the_wrong_family(trainer_cls):
    """The remaining flat vector-field config must not fall through as callable."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.raises(TypeError):
        trainer_cls(
            prior,
            density_estimator=VectorFieldEstimatorBuilder(),
            show_progress_bars=False,
        )


@pytest.mark.parametrize("trainer_cls", [NPE_C, NLE_A], ids=["npe", "nle"])
@pytest.mark.parametrize(
    "density_estimator", [TabPFNConfig(), "tabpfn"], ids=["config", "string"]
)
def test_gradient_trainer_rejects_tabpfn(trainer_cls, density_estimator):
    """TabPFNFlow has no loss and is supported only by training-free NPE_PFN."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.raises(TypeError, match="Use NPE_PFN instead"):
        trainer_cls(
            prior,
            density_estimator=density_estimator,
            show_progress_bars=False,
        )


class _FrozenDensityEstimator(ConditionalDensityEstimator):
    """Well-formed estimator with nothing to fit, like a pretrained wrapper."""

    def __init__(self):
        net = torch.nn.Linear(3, 2)
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        super().__init__(net=net, input_shape=(2,), condition_shape=(3,))

    def log_prob(self, input, condition, **kwargs):
        return zeros(input.shape[:2])

    def loss(self, input, condition, **kwargs):
        return zeros(input.shape[0])

    def sample(self, sample_shape, condition, **kwargs):
        return zeros(*sample_shape, condition.shape[0], 2)


@pytest.mark.parametrize("trainer_cls", [NPE_C, NLE_A], ids=["npe", "nle"])
def test_gradient_trainer_rejects_estimators_with_nothing_to_fit(trainer_cls):
    """Callable builders bypass the config checks, so the shared training
    loop rejects the built estimator before the optimizer is created."""
    prior = MultivariateNormal(zeros(2), eye(2))
    inference = trainer_cls(
        prior,
        density_estimator=lambda *_: _FrozenDensityEstimator(),
        show_progress_bars=False,
    )
    theta, x = prior.sample((10,)), torch.randn(10, 3)
    inference.append_simulations(theta, x)

    with pytest.raises(TypeError, match="no trainable parameters"):
        inference.train(max_num_epochs=1)


def test_npe_pfn_accepts_tabpfn_config():
    prior = MultivariateNormal(zeros(2), eye(2))
    trainer = NPE_PFN(
        prior,
        density_estimator=TabPFNConfig(),
        show_progress_bars=False,
    )

    assert callable(trainer._build_neural_net)


@pytest.mark.parametrize("trainer_cls", [MNPE, MNLE], ids=["mnpe", "mnle"])
def test_mixed_trainer_rejects_plain_density_config(trainer_cls):
    """MNPE/MNLE require the mixed config, not a plain density config."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.raises(TypeError, match="MixedConfig"):
        trainer_cls(prior, density_estimator=MAFConfig())


def test_npe_a_default_does_not_warn():
    """NPE_A's default must not trigger the string deprecation."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        NPE_A(prior)


def test_npe_a_string_warns():
    """The one NPE_A string stays available, deprecated."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.warns(FutureWarning, match="MDNConfig"):
        NPE_A(prior, density_estimator="mdn_snpe_a")


@pytest.mark.parametrize("num_components", [3, 7], ids=["three", "seven"])
def test_npe_a_binds_num_components_at_initialization(num_components):
    """The trainer's component count is bound once during initialization."""
    prior = MultivariateNormal(zeros(2), eye(2))
    theta = prior.sample((20,))
    x = theta + 0.1 * torch.randn_like(theta)

    trainer = NPE_A(
        prior,
        density_estimator=MDNConfig(hidden_features=16),
        num_components=num_components,
        show_progress_bars=False,
    )
    estimator = trainer._build_neural_net(theta, x)

    assert estimator.net._num_components == num_components


def test_npe_a_rejects_a_conflicting_num_components():
    """Setting it on both sides is ambiguous, so it must not be guessed."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.raises(ValueError, match="num_components"):
        NPE_A(prior, density_estimator=MDNConfig(num_components=5), num_components=20)


def test_npe_a_rejects_a_non_mdn_config():
    """NPE-A's analytical correction needs a mixture of Gaussians."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.raises(TypeError, match="MDNConfig"):
        NPE_A(prior, density_estimator=NSFConfig())
