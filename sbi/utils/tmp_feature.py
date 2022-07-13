import torch
import numpy as np
from scipy.stats import gaussian_kde
from sbi.utils.plot import _get_default_opts, _update, ensure_numpy


def _get_limits(samples, limits=None):

    if type(samples) != list:
        samples = ensure_numpy(samples)
        samples = [samples]
    else:
        for i, sample_pack in enumerate(samples):
            samples[i] = ensure_numpy(samples[i])

    # Dimensionality of the problem.
    dim = samples[0].shape[1]

    if limits == [] or limits is None:
        limits = []
        for d in range(dim):
            min = +np.inf
            max = -np.inf
            for sample in samples:
                min_ = sample[:, d].min()
                min = min_ if min_ < min else min
                max_ = sample[:, d].max()
                max = max_ if max_ > max else max
            limits.append([min, max])
    else:
        if len(limits) == 1:
            limits = [limits[0] for _ in range(dim)]
        else:
            limits = limits
    limits = torch.as_tensor(limits)

    return limits


def posterior_peaks(samples, return_dict=False, **kwargs):
    '''
    Finds the peaks of the posterior distribution.

    Args:
        samples: torch.tensor, samples from posterior
    Returns: torch.tensor, peaks of the posterior distribution
            if labels provided as a list of strings, and return_dict is True
            returns a dictionary of peaks

    '''

    opts = _get_default_opts()
    opts = _update(opts, kwargs)

    limits = _get_limits(samples)
    samples = samples.numpy()
    n, dim = samples.shape

    try:
        labels = opts['labels']
    except:
        labels = range(dim)

    peaks = {}
    if labels is None:
        labels = range(dim)
    for i in range(dim):
        peaks[labels[i]] = 0

    for row in range(dim):
        density = gaussian_kde(
            samples[:, row],
            bw_method=opts["kde_diag"]["bw_method"])
        xs = np.linspace(
            limits[row, 0], limits[row, 1],
            opts["kde_diag"]["bins"])
        ys = density(xs)

        # y, x = np.histogram(samples[:, row], bins=bins)
        peaks[labels[row]] = xs[ys.argmax()]

    if return_dict:
        return peaks
    else:
        return list(peaks.values())

if __name__ == "__main__":

    '''
    usage example:
    '''
    import torch
    from sbi import utils as utils
    from sbi import analysis as analysis
    from sbi.inference.base import infer

    num_dim = 3
    prior = utils.BoxUniform(low=-2*torch.ones(num_dim), high=2*torch.ones(num_dim))

    def simulator(parameter_set):
        return 1.0 + parameter_set + torch.randn(parameter_set.shape) * 0.1
    
    posterior = infer(simulator, prior, method='SNPE', num_simulations=1000)
    observation = torch.zeros(3)

    samples = posterior.sample((10000,), x=observation)
    log_probability = posterior.log_prob(samples, x=observation)
    _ = analysis.pairplot(samples, limits=[[-2,2],[-2,2],[-2,2]], figsize=(6,6))

    peaks = posterior_peaks(samples, labels=['x', 'y', 'z'], return_dict=True)
    print(peaks)
