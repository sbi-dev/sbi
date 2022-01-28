import torch

from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

from joblib import Parallel, delayed
from sbi.inference import infer


def test_posterior_ensemble():

    # train ensemble components
    ensemble_size = 10
    posteriors = Parallel(n_jobs=-1)(
        delayed(infer)(prior, simulator, num_samples) for i in range(ensemble_size)
    )

    # create ensemble
    ensemble = NeuralPosteriorEnsemble(posteriors)
    ensemble.set_default_x(torch.zeros((3,)))

    # test log_prob
    ensemble.log_prob()
    assert something

    # test sample
    ensemble.sample((1,))
    assert something

    # test map
    ensemble.map()
    assert something
