# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import threading

import pytest
import torch
from torch.distributions import MultivariateNormal
from tqdm.auto import tqdm

from sbi.inference.posteriors import VectorFieldPosterior
from sbi.neural_nets import posterior_score_nn
from sbi.samplers.importance import sir
from sbi.samplers.importance.importance_sampling import importance_sample
from sbi.samplers.rejection import rejection
from sbi.samplers.score import diffuser
from sbi.utils import BoxUniform
from sbi.utils.pbar import is_nested, nested_pbar_context


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

    for module in (rejection, sir, diffuser):
        monkeypatch.setattr(module, "tqdm", RecordingTqdm)
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
@pytest.mark.parametrize("reject_outside_prior", [True, False])
@pytest.mark.parametrize("show_progress_bars", [True, False])
def test_vector_field_posterior_shows_at_most_one_bar(
    reject_outside_prior, show_progress_bars, recorded_bars
):
    """Regression test for #1811.

    With rejection sampling, only the rejection bar is shown. Without it, the
    diffusion bar must stay visible because it is the only one.
    """
    num_dim = 2
    prior = BoxUniform(-3 * torch.ones(num_dim), 3 * torch.ones(num_dim))
    theta = prior.sample((200,))
    x = theta + 0.1 * torch.randn_like(theta)
    estimator = posterior_score_nn(sde_type="vp")(theta, x)
    posterior = VectorFieldPosterior(vector_field_estimator=estimator, prior=prior)

    posterior.sample(
        (10,),
        x=x[:1],
        steps=3,
        show_progress_bars=show_progress_bars,
        reject_outside_prior=reject_outside_prior,
    )

    bars = shown(recorded_bars)
    if not show_progress_bars:
        assert bars == []
    elif reject_outside_prior:
        assert len(bars) == 1 and bars[0].startswith("Drawing")
    else:
        assert len(bars) == 1 and bars[0].startswith("Generating")
