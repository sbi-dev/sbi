# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import threading

import pytest
import torch
from torch.distributions import MultivariateNormal
from tqdm.auto import tqdm

from sbi.inference.posteriors import VectorFieldPosterior
from sbi.inference.posteriors.mcmc_posterior import build_from_potential
from sbi.neural_nets import posterior_score_nn
from sbi.samplers.importance import sir
from sbi.samplers.importance.importance_sampling import importance_sample
from sbi.samplers.mcmc import slice_numpy
from sbi.samplers.rejection import rejection
from sbi.samplers.score import diffuser
from sbi.utils import BoxUniform
from sbi.utils.pbar import is_nested, nested_pbar_context, sampling_desc


class TestNestedPbarContext:
    """Unit tests for the thread-local nesting counter."""

    def test_not_nested_by_default(self):
        assert not is_nested()

    def test_nested_inside_context(self):
        with nested_pbar_context():
            assert is_nested()

    def test_not_nested_after_context_exit(self):
        with nested_pbar_context():
            pass
        assert not is_nested()

    def test_not_nested_after_exception(self):
        with pytest.raises(RuntimeError), nested_pbar_context():
            raise RuntimeError
        assert not is_nested()

    def test_deeply_nested_contexts(self):
        with nested_pbar_context():
            assert is_nested()
            with nested_pbar_context():
                assert is_nested()
                with nested_pbar_context():
                    assert is_nested()
                assert is_nested()
            assert is_nested()
        assert not is_nested()

    def test_thread_isolation(self):
        main_sees = []
        worker_sees = []
        barrier = threading.Barrier(2, timeout=5)

        def worker():
            with nested_pbar_context():
                barrier.wait()
                worker_sees.append(is_nested())
                barrier.wait()

        t = threading.Thread(target=worker)
        t.start()
        barrier.wait()
        main_sees.append(is_nested())
        barrier.wait()
        t.join()

        assert not main_sees[0], "main should not see worker's context"
        assert worker_sees[0], "worker should see its own context"


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            dict(num_samples=1000, method="rejection"),
            "Drawing 1000 samples [rejection]",
        ),
        (
            dict(num_samples=1, method="rejection", num_xos=5),
            "Drawing 1 sample for each of 5 observations [rejection]",
        ),
        (
            dict(num_samples=100, method="slice_np", num_chains=20, num_workers=1),
            "Drawing 100 samples [slice_np, 20 chains, 1 worker]",
        ),
        (
            dict(num_samples=100, method="sde", num_steps=499),
            "Drawing 100 samples [sde, 499 steps]",
        ),
    ],
)
def test_sampling_desc(kwargs, expected):
    assert sampling_desc(**kwargs) == expected


@pytest.fixture
def recorded_bars(monkeypatch):
    """Records `(desc, disable)` of every progress bar the samplers create.

    The recorded bars are silenced so that the test output stays clean.
    """
    records = []

    class RecordingTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            records.append((kwargs.get("desc", ""), kwargs.get("disable", False)))
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    for module in (rejection, sir, diffuser, slice_numpy):
        monkeypatch.setattr(module, "tqdm", RecordingTqdm)
    monkeypatch.setattr(
        slice_numpy, "trange", lambda n, **kwargs: RecordingTqdm(range(n), **kwargs)
    )
    return records


def shown(records):
    """Returns the descriptions of the recorded bars that were not disabled."""
    return [desc for desc, disable in records if not disable]


class RecordingProposal:
    """Proposal that records `is_nested()` at every `sample()` call."""

    def __init__(self, distribution):
        self.distribution = distribution
        self.nested_at_sample = []

    def sample(self, sample_shape, **kwargs):
        self.nested_at_sample.append(is_nested())
        return self.distribution.sample(torch.Size(sample_shape))

    def log_prob(self, theta, **kwargs):
        return self.distribution.log_prob(theta)


def _run_accept_reject_sample(proposal, potential_fn):
    return rejection.accept_reject_sample(
        proposal=proposal.sample,
        accept_reject_fn=lambda theta: torch.ones(theta.shape[0], dtype=torch.bool),
        num_samples=5,
        show_progress_bars=True,
    )


def _run_rejection_sample(proposal, potential_fn):
    return rejection.rejection_sample(
        potential_fn,
        proposal,
        num_samples=5,
        num_samples_to_find_max=10,
        num_iter_to_find_max=1,
        show_progress_bars=True,
    )


def _run_sir(proposal, potential_fn):
    return sir.sampling_importance_resampling(
        potential_fn,
        proposal,
        num_samples=5,
        num_candidate_samples=2,
        show_progress_bars=True,
    )


@pytest.mark.parametrize(
    "run_sampler", [_run_accept_reject_sample, _run_rejection_sample, _run_sir]
)
def test_sampler_nests_proposal_calls_and_shows_one_bar(run_sampler, recorded_bars):
    """Every proposal call runs nested, and only the outermost sampler shows a bar."""
    proposal = RecordingProposal(MultivariateNormal(torch.zeros(2), torch.eye(2)))

    def potential_fn(theta):
        return -0.5 * (theta**2).sum(-1)

    run_sampler(proposal, potential_fn)
    assert proposal.nested_at_sample, "proposal was not called"
    assert all(proposal.nested_at_sample), "a proposal call ran outside the context"
    assert len(shown(recorded_bars)) == 1

    recorded_bars.clear()
    with nested_pbar_context():
        run_sampler(proposal, potential_fn)
    assert shown(recorded_bars) == [], "a nested sampler must not show a bar"


@pytest.mark.parametrize("show_progress_bars", [True, False])
def test_importance_sample_shows_proposal_bar_only_on_request(show_progress_bars):
    """`importance_sample` has no bar of its own. The bar of the proposal, if it has
    one, is shown only if progress bars are requested."""
    proposal = RecordingProposal(MultivariateNormal(torch.zeros(2), torch.eye(2)))

    importance_sample(
        lambda theta: -0.5 * (theta**2).sum(-1),
        proposal,
        num_samples=5,
        show_progress_bars=show_progress_bars,
    )

    assert proposal.nested_at_sample == [not show_progress_bars]


@pytest.mark.filterwarnings("ignore:.*lie outside the prior support")
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("reject_outside_prior", [True, False])
@pytest.mark.parametrize("show_progress_bars", [True, False])
def test_vector_field_posterior_shows_at_most_one_bar(
    batched, reject_outside_prior, show_progress_bars, recorded_bars
):
    """Regression test for #1811, for `sample()` and `sample_batched()`.

    With rejection sampling, only the rejection bar is shown. Without it, the
    diffusion bar must stay visible because it is the only one.
    """
    num_dim = 2
    prior = BoxUniform(-3 * torch.ones(num_dim), 3 * torch.ones(num_dim))
    theta = prior.sample((200,))
    x = theta + 0.1 * torch.randn_like(theta)
    estimator = posterior_score_nn(sde_type="vp")(theta, x)
    posterior = VectorFieldPosterior(vector_field_estimator=estimator, prior=prior)

    sample_fn = posterior.sample_batched if batched else posterior.sample
    sample_fn(
        (10,),
        x=x[:2] if batched else x[:1],
        steps=3,
        show_progress_bars=show_progress_bars,
        reject_outside_prior=reject_outside_prior,
    )

    bars = shown(recorded_bars)
    if not show_progress_bars:
        assert bars == []
        return
    assert len(bars) == 1
    assert ("[rejection]" in bars[0]) == reject_outside_prior
    assert ("for each of 2 observations" in bars[0]) == batched


def _gaussian_mcmc_posterior():
    prior = BoxUniform(-2 * torch.ones(2), 2 * torch.ones(2))

    def potential_fn(theta, x):
        return -x * (theta**2).sum(axis=-1)

    return build_from_potential(potential_fn, prior, x=torch.tensor([0.5]))


@pytest.mark.mcmc
def test_serial_slice_sampler_shows_one_bar_with_one_worker(recorded_bars):
    """The chains must not show bars of their own, one or two per chain."""
    _gaussian_mcmc_posterior().sample(
        (3,),
        method="slice_np",
        num_chains=2,
        num_workers=1,
        warmup_steps=2,
        thin=1,
        show_progress_bars=True,
    )

    assert shown(recorded_bars) == ["Drawing 3 samples [slice_np, 2 chains, 1 worker]"]


@pytest.mark.mcmc
def test_batched_mcmc_bar_counts_per_observation(recorded_bars):
    """Samples and chains in the bar are counted per observation, not in total."""
    _gaussian_mcmc_posterior().sample_batched(
        (3,),
        x=torch.tensor([[0.2], [0.8]]),
        method="slice_np_vectorized",
        num_chains=2,
        warmup_steps=2,
        thin=1,
        show_progress_bars=True,
    )

    assert shown(recorded_bars) == [
        "Drawing 3 samples for each of 2 observations [slice_np_vectorized, 2 chains]"
    ]
